import org.apache.spark.sql.SparkSession
import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, StandardScaler, VectorAssembler, UnivariateFeatureSelector, PCA}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.ml.stat._
import org.apache.spark.sql._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.evaluation.RegressionMetrics
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter
import java.io.File


object MyApp {
  // Outlier detection
  def withMahalanobis(df: DataFrame, inputCol: String, k: Int): DataFrame = {
    val Row(coeff1: Matrix) = Correlation.corr(df, inputCol).head

    val invCovariance = inv(new breeze.linalg.DenseMatrix(k, k, coeff1.toArray))

    val mahalanobis = udf[Double, Vector] { v =>
      val vB = DenseVector(v.toArray)
      vB.t * invCovariance * vB
    }

    df.withColumn("mahalanobis", mahalanobis(df(inputCol)))
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    import spark.implicits._

    // Set schema of the dataframe
    val schema = StructType(Array(
      StructField("Year", IntegerType, nullable = true),
      StructField("Month", IntegerType, nullable = true),
      StructField("DayofMonth", IntegerType, nullable = true),
      StructField("DayOfWeek", IntegerType, nullable = true),
      StructField("DepTime", IntegerType, nullable = true),
      StructField("CRSDepTime", IntegerType, nullable = true),
      StructField("ArrTime", StringType, nullable = true),
      StructField("CRSArrTime", IntegerType, nullable = true),
      StructField("UniqueCarrier", StringType, nullable = true),
      StructField("FlightNum", IntegerType, nullable = true),
      StructField("TailNum", IntegerType, nullable = true),
      StructField("ActualElapsedTime", StringType, nullable = true),
      StructField("CRSElapsedTime", IntegerType, nullable = true),
      StructField("AirTime", StringType, nullable = true),
      StructField("ArrDelay", DoubleType, nullable = true),
      StructField("DepDelay", IntegerType, nullable = true),
      StructField("Origin", StringType, nullable = true),
      StructField("Dest", StringType, nullable = true),
      StructField("Distance", IntegerType, nullable = true),
      StructField("TaxiIn", StringType, nullable = true),
      StructField("TaxiOut", IntegerType, nullable = true),
      StructField("Cancelled", IntegerType, nullable = true),
      StructField("CancellationCode", StringType, nullable = true),
      StructField("Diverted", StringType, nullable = true),
      StructField("CarrierDelay", StringType, nullable = true),
      StructField("WeatherDelay", StringType, nullable = true),
      StructField("NASDelay", StringType, nullable = true),
      StructField("SecurityDelay", StringType, nullable = true),
      StructField("LateAircraftDelay", StringType, nullable = true)
    ))

    // Transform variables with format hhmm to minutes after 00:00
    val parseTime = udf((s: Int) => s % 100 + (s / 100) * 60)


    // Load csv files selected by the user, drop unuseful variables and rows with a null value for the target variable and transform time variables
    var filePaths = Array.empty[File]
    val chooser = new JFileChooser()
    val extension_filter = new FileNameExtensionFilter("CSV files", "csv"); // Show only csv files
    chooser.setCurrentDirectory(new java.io.File("."))
    chooser.setDialogTitle("Select CSV files")
    chooser.setFileFilter(extension_filter)
    chooser.setMultiSelectionEnabled(true); // Allow multiple files selection
    while (filePaths.isEmpty) {
      if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
        filePaths = chooser.getSelectedFiles
      }
    }
    val df = spark.read.option("header", "true")
      .schema(schema)
      .csv(filePaths.map(_.getAbsolutePath): _*)
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "Cancelled", "CancellationCode", "TailNum")
      .withColumn("DepTime", parseTime($"DepTime"))
      .withColumn("CRSDepTime", parseTime($"CRSDepTime"))
      .withColumn("CRSArrTime", parseTime($"CRSArrTime"))
      .na.drop()

    // Check null values of each variable
    for (c <- df.columns) {
      printf("Column %s: %d null values\n", c, df.filter(col(c).isNull || col(c) === "NA").count())
    }

    // Transform categorical variables
    val indexer_city = new StringIndexer()
      .setInputCols(Array("Origin", "Dest"))
      .setOutputCols(Array("Origin_cat", "Dest_cat"))
    val indexer_carrier = new StringIndexer()
      .setInputCol("UniqueCarrier")
      .setOutputCol("UniqueCarrier_cat")

    // Assemble features
    val features_names = df.columns.toSet -- Set("ArrDelay", "UniqueCarrier", "Origin", "Dest") ++ Set("UniqueCarrier_cat", "Origin_cat", "Dest_cat")
    val assembler = new VectorAssembler()
      .setInputCols(features_names.toArray)
      .setOutputCol("features")

    // Scale features
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true).setWithMean(true)

    // Perform the Univariate Filter FSS
    val selector = new UnivariateFeatureSelector()
      .setFeatureType("continuous")
      .setLabelType("continuous")
      .setSelectionMode("numTopFeatures")
      .setSelectionThreshold(1)
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("ArrDelay")
      .setOutputCol("selectedFeatures")

    // Create a pipeline for the defined transformations and perform them
    val pipeline = new Pipeline()
      .setStages(Array(indexer_city, indexer_carrier, assembler, scaler))
    val dfTransformed = pipeline.fit(df).transform(df)

    // Show results
    dfTransformed.show()

    // Divide data into training and testing for transformed dataframe 1
    val split = dfTransformed.randomSplit(Array(0.7, 0.3))
    val training = split(0)
    val test = split(1)

    val K = 3
    val PCAdf = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pca-features")
      .setK(K).fit(training)
      .transform(training)
    // PCAdf.select(col("pca-features")).show()
    val mahalanobis: DataFrame = withMahalanobis(PCAdf, "pca-features", K)
    mahalanobis.select(col("mahalanobis")).show()
    val quantiles = mahalanobis.stat.approxQuantile("mahalanobis", Array(0.25, 0.75), 0.0)
    PCAdf.select(col("pca-features")).show()
    val Q1 = quantiles(0)
    val Q3 = quantiles(1)
    val IQR = Q3 - Q1
    val upperRange = Q3 + 1.5 * IQR
    val cleanTrainingDF = mahalanobis.filter($"mahalanobis" < upperRange)

    //val rdd = cleanTrainingDF.select(col("scaledFeatures")).rdd.map(list)
    //val rowmatrix = RowMatrix(rdd)

    // MACHINE LEARNING MODELS
    val lr = new LinearRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    val pipeline11 = new Pipeline()
      .setStages(Array(lr))

    val lrModel = pipeline11.fit(cleanTrainingDF).transform(test)

    val predictions = lrModel.select("prediction").rdd.map(_.getDouble(0))
    val labels = lrModel.select("ArrDelay").rdd.map(_.getDouble(0))
    val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    val Rsquare = new RegressionMetrics(predictions.zip(labels)).r2
    println(s"Root mean squared error (RMSE) for linear regression: $RMSE")
    println(s"R-square for linear regression: $Rsquare")

    val glr = new GeneralizedLinearRegression()
      .setFamily("poisson")
      .setLink("log")
      .setMaxIter(10)
      .setRegParam(0.3)
      .setLabelCol("ArrDelay")
      .setFeaturesCol("scaledFeatures")
    val pipeline33 = new Pipeline().setStages(Array(lr))
    val glrModel = pipeline33.fit(training).transform(test)

    val predictions33 = glrModel.select("prediction").rdd.map(_.getDouble(0))
    val labels33 = glrModel.select("ArrDelay").rdd.map(_.getDouble(0))
    val RMSE33 = new RegressionMetrics(predictions33.zip(labels33)).rootMeanSquaredError
    val Rsquare33 = new RegressionMetrics(predictions33.zip(labels33)).r2
    println(s"  Root mean squared error (RMSE) for GLR: $RMSE33")
    println(s"  R-square for GLR: $Rsquare33")


  }
}
