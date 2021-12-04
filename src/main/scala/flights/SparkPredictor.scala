package flights

import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PCA, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import java.io.{File, PrintWriter}

class SparkPredictor {
  // TODO: tune SparkConf
  val spark: SparkSession = SparkSession.builder().getOrCreate()

  import spark.implicits._

  var checking = false
  var trainFiles = Array.empty[File]
  var testFiles = Array.empty[File]
  var testPercentage = 0.0
  val schema: StructType = StructType(Array(
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

  /**
   * Adds to a DataFrame one column that represents the Mahalanobis distance to the mean for each row
   *
   * @param df       a DataFrame
   * @param inputCol name of the column where the features are stored
   * @param k        number of features stored in `inputCol`
   * @return a DataFrame with the new column added
   */
  def addMahalanobis(df: DataFrame, inputCol: String, k: Int): DataFrame = {
    val Row(coeff1: Matrix) = Correlation.corr(df, inputCol).head
    val invCovariance = inv(new breeze.linalg.DenseMatrix(k, k, coeff1.toArray))
    val mahalanobis = udf[Double, Vector] { v =>
      val vB = DenseVector(v.toArray)
      vB.t * invCovariance * vB
    }

    df.withColumn("mahalanobis", mahalanobis(df(inputCol)))
  }

  /**
   * Loads data from the corresponding files and performs some basic transformations
   *
   * @param testing  a boolean to know if the files to load are for testing
   * @param checking a boolean to know if the data to load is for previous checks
   * @return a DataFrame obtained after reading and transforming the file
   */
  def loadData(testing: Boolean, checking: Boolean): DataFrame = {
    var files = trainFiles
    if (testing)
      files = testFiles

    // TODO: check files are correct

    // Transform variables with format hhmm to minutes after 00:00
    val parseTime: UserDefinedFunction = udf((s: Int) => s % 100 + (s / 100) * 60)
    val df = spark.read.option("header", "true")
      .schema(schema)
      .csv(files.map(_.getAbsolutePath): _*)
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay",
        "NASDelay", "SecurityDelay", "LateAircraftDelay", "Cancelled", "CancellationCode", "TailNum")
      .withColumn("DepTime", parseTime($"DepTime"))
      .withColumn("CRSDepTime", parseTime($"CRSDepTime"))
      .withColumn("CRSArrTime", parseTime($"CRSArrTime"))
      .na.drop()

    if (checking)
      return df
    df.select("CRSDepTime", "CRSArrTime", "DepTime", "TaxiOut", "DepDelay", "ArrDelay", "Month", "Year")

    // TODO: transform times to the same time zone
    // TODO: create new variables, like number of flights that get to the same airport at the same time
  }

  /**
   * Creates and applies a regression model with the training and testing DataFrames received
   *
   * @param regType type of regression
   * @param dfTrain training DataFrame
   * @param dfTest  testing DataFrame
   * @return a tuple with root mean squared error, R square, adjusted R square and a representation of 20 predictions
   */
  def regressionModel(regType: String, dfTrain: DataFrame, dfTest: DataFrame): (Double, Double, Double, String) = {
    var pos = 0
    val regTypes = Array(new LinearRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8),
      new DecisionTreeRegressor()
        .setFeaturesCol("scaledFeatures")
        .setLabelCol("ArrDelay"))
    if (regType == "decision tree")
      pos = 1

    val pipeline = new Pipeline().setStages(Array(regTypes(pos)))
    val model = pipeline.fit(dfTrain).transform(dfTest)
    val predictions = model.select("prediction").rdd.map(_.getDouble(0))
    val labels = model.select("ArrDelay").rdd.map(_.getDouble(0))
    val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    val Rsquare = new RegressionMetrics(predictions.zip(labels)).r2
    val n = dfTrain.count
    val k = 7
    val Rsquare_adjusted = 1 - ((1 - Rsquare) * (n - 1)) / (n - k - 1)

    (RMSE, Rsquare, Rsquare_adjusted, model.select("ArrDelay", "prediction").take(20).mkString("\n"))
  }

  /**
   * Executes the predictor according to the values of the attributes
   */
  def run(): Unit = {
    // TODO: make a better use of pipelines
    val df = loadData(testing = false, checking = checking)
    // Assemble features
    var features_names = df.columns.toSet -- Set("ArrDelay")
    if (checking)
      features_names = df.columns.toSet ++ Set("Origin_cat", "Dest_cat", "UniqueCarrier_cat") --
        Set("ArrDelay", "Origin", "Dest", "UniqueCarrier")
    val assembler = new VectorAssembler()
      .setInputCols(features_names.toArray)
      .setOutputCol("features")

    // Scale features
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true).setWithMean(true)

    var pipeline = new Pipeline()
      .setStages(Array(assembler, scaler))

    val outFile = new File("output.txt")
    val output = new PrintWriter(outFile)
    output.print("Execution summary:\n\n")

    if (checking) {
      //      for (c <- df.columns)
      //        output.printf("Column %s: %s null values\n", c, df.filter(col(c).isNull || col(c) === "NA").count().toString)

      val indexer_city = new StringIndexer()
        .setInputCols(Array("Origin", "Dest"))
        .setOutputCols(Array("Origin_cat", "Dest_cat"))
      val indexer_carrier = new StringIndexer()
        .setInputCol("UniqueCarrier")
        .setOutputCol("UniqueCarrier_cat")
      val assemblerArrDelay = new VectorAssembler()
        .setInputCols((features_names ++ Set("ArrDelay")).toArray)
        .setOutputCol("featuresCorr")
      val scalerArrDelay = new StandardScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeaturesCorr")
        .setWithStd(true).setWithMean(true)
      pipeline = new Pipeline()
        .setStages(Array(indexer_city, indexer_carrier, assembler, scaler, assemblerArrDelay, scalerArrDelay))
    }

    val pipelineModel = pipeline.fit(df)
    val dfScaled = pipelineModel.transform(df)

    // Divide data into training and testing
    val split = dfScaled.randomSplit(Array(1 - testPercentage, testPercentage))
    val training = split(0)
    var dfTest = split(1)

    if (testPercentage == 0)
      dfTest = pipelineModel.transform(loadData(testing = true, checking = checking))

    // TODO: check the requirements for the input file in order to avoid PCA and correlation matrix throwing an exception
    // Perform Principal Component Analysis and delete outliers from the training set
    val k = 3
    val dfPCA = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pca-features")
      .setK(k).fit(training)
      .transform(training)

    val dfMahalanobis: DataFrame = addMahalanobis(dfPCA, "pca-features", k)
    val quantiles = dfMahalanobis.stat.approxQuantile("mahalanobis", Array(0.25, 0.75), 0.0)
    val upperRange = quantiles(1) + 1.5 * (quantiles(1) - quantiles(0))
    val dfTraining = dfMahalanobis.filter($"mahalanobis" < upperRange)

    if (checking) {
      // TODO: round values of column `pearson(scaledFeaturesCorr)` in dfCorr to have a better print
      val dfCorr = Correlation.corr(dfTraining, "scaledFeaturesCorr")
      val Row(coeff: Matrix) = dfCorr.head
      output.println(s"\nPearson correlation matrix:\n  " + coeff.toString(16, 1000))
    }

    val regressors = Array("linear", "decision tree")
    for (reg <- regressors) {
      val (rmse, rSquare, rSquareAdj, predictions) = regressionModel(reg, dfTraining, dfTest)
      output.println("\nRESULTS FOR " + reg.toUpperCase + " REGRESSION")
      output.println(s"  - Root mean squared error (RMSE): $rmse")
      output.println(s"  - R-square: $rSquare")
      output.println(s"  - Adjusted R-square: $rSquareAdj")
      output.println("\n  Examples of predictions and actual arrival delays. Format: [ArrDelay, Prediction]")
      output.println(predictions)
    }
    output.close()
  }
}
