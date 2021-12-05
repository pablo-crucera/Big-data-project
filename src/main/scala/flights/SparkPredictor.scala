package flights

import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PCA, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, LinearRegression}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.io.{File, PrintWriter}
import scala.collection.mutable.ArrayBuffer

class SparkPredictor {
  val spark: SparkSession = SparkSession.builder().getOrCreate()

  import spark.implicits._

  var checking = false
  var trainFiles = Array.empty[File]
  var testFiles = Array.empty[File]
  var testPercentage = 0.3
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
  val checkCols: Array[String] = Array("CRSDepTime", "CRSArrTime", "DepTime", "TaxiOut", "DepDelay", "ArrDelay",
    "Month", "Year", "DayOfMonth", "DayOfWeek", "UniqueCarrier", "FlightNum",
    "CRSElapsedTime", "Origin", "Dest", "Distance")
  val finalCols: Array[String] = Array("CRSDepTime", "CRSArrTime", "DepTime", "TaxiOut", "DepDelay", "ArrDelay",
    "Month", "Year")

  /**
   * Adds to a DataFrame one column that represents the Mahalanobis distance to the mean for each row
   *
   * @param df       a DataFrame
   * @param inputCol name of the column where the features are stored
   * @param k        number of features stored in `inputCol`
   * @return a DataFrame with the new column added
   */
  def addMahalanobis(df: DataFrame, inputCol: String, k: Int): DataFrame = {
    val Row(corrM: Matrix) = Correlation.corr(df, inputCol).head
    val invCovariance = inv(new breeze.linalg.DenseMatrix(k, k, corrM.toArray))
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
      .select(checkCols.map(s => col(s)): _*)
      .withColumn("DepTime", parseTime($"DepTime"))
      .withColumn("CRSDepTime", parseTime($"CRSDepTime"))
      .withColumn("CRSArrTime", parseTime($"CRSArrTime"))
      .na.drop()

    if (checking)
      return df.filter($"Origin" =!= "NA" && $"Dest" =!= "NA" && $"UniqueCarrier" =!= "NA")
    df.select(finalCols.map(s => col(s)): _*)

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
    val regTypes = Map("linear" -> new LinearRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8),
      "decision tree" -> new DecisionTreeRegressor()
        .setFeaturesCol("scaledFeatures")
        .setLabelCol("ArrDelay"))

    val pipeline = new Pipeline().setStages(Array(regTypes(regType)))
    val model = pipeline.fit(dfTrain).transform(dfTest)
    val predictions = model.select("prediction").rdd.map(_.getDouble(0))
    val labels = model.select("ArrDelay").rdd.map(_.getDouble(0))
    val rmse = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    val rSquare = new RegressionMetrics(predictions.zip(labels)).r2
    val n = dfTrain.count
    val k = 7
    val rSquareAdj = 1 - ((1 - rSquare) * (n - 1)) / (n - k - 1)

    (rmse, rSquare, rSquareAdj, model.select("ArrDelay", "prediction").take(20).mkString("\n"))
  }

  /**
   * Gets a string representation of the correlation matrix
   *
   * @param df the DataFrame to calculate the correlation matrix from. It must have a column named `scaledFeaturesCorr`
   * @param colsNames names of the columns of the matrix
   */
  def getCorrMString(df: DataFrame, colsNames: ArrayBuffer[String]): String = {
    val Row(corrM: Matrix) = Correlation.corr(df, "scaledFeaturesCorr").head
    val corrMArr = corrM.toArray.map(e => f"$e%.2f")
    var result = ""
    result += "Pearson correlation matrix:\n"
    result += "Columns order: " + colsNames.mkString(", ") + "\n"
    for (i <- 0 until (corrMArr.length / corrM.numCols)) {
      for (j <- 0 until corrM.numCols)
        result += " " * (5 - corrMArr(i * corrM.numCols + j).length) + corrMArr(i * corrM.numCols + j) + " "
      result += "\n"
    }
    result
  }

  /**
   * Prints a summary of the execution performed in a file called `output.txt`
   *
   * @param resultsLinear Results for linear regression
   * @param resultsTree Results for decision tree regression
   * @param df the Dataframe to calculate the correlation matrix from
   * @param featuresNames Names of the features in the correlation matrix
   */
  def printSummary(resultsLinear: (Double, Double, Double, String), resultsTree: (Double, Double, Double, String),
                   df:DataFrame, featuresNames: ArrayBuffer[String]): Unit = {
    val outFile = new File("output.txt")
    val output = new PrintWriter(outFile)

    output.print("Execution summary:\n\n")
    if (checking)
      output.println(getCorrMString(df, featuresNames))

    val text =
      """
        |RESULTS FOR %s REGRESSION
        |   - Root mean squared error (RMSE): %f
        |   - R-square: %f
        |   - Adjusted R-square: %f
        |
        |   - Examples of predictions and actual arrival delays. Format: [ArrDelay, Prediction]
        |%s
        |""".stripMargin

    output.println(text.format("LINEAR", resultsLinear._1, resultsLinear._2, resultsLinear._3, resultsLinear._4))
    output.println(text.format("DECISION TREE", resultsTree._1, resultsTree._2, resultsTree._3, resultsTree._4))
    output.close()
  }

  /**
   * Executes the predictor according to the values of the attributes
   */
  def run(): Unit = {
    // TODO: make a better use of pipelines
    val df = loadData(testing = false, checking = checking)

    var featuresNames = ArrayBuffer.empty[String]
    if (checking)
      featuresNames = checkCols.to[ArrayBuffer] -= "ArrDelay" -= "Origin" -= "Dest" -= "UniqueCarrier" +=
        "Origin_cat" += "Dest_cat" += "UniqueCarrier_cat"
    else
      featuresNames = finalCols.to[ArrayBuffer] -= "ArrDelay"

    // Assemble features
    val assembler = new VectorAssembler()
      .setInputCols(featuresNames.toArray)
      .setOutputCol("features")

    // Scale features
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true).setWithMean(true)

    var pipeline = new Pipeline()
      .setStages(Array(assembler, scaler))

    featuresNames += "ArrDelay"
    // Add more information if we are checking
    if (checking) {
      val indexerCity = new StringIndexer()
        .setInputCols(Array("Origin", "Dest"))
        .setOutputCols(Array("Origin_cat", "Dest_cat"))
      val indexerCarrier = new StringIndexer()
        .setInputCol("UniqueCarrier")
        .setOutputCol("UniqueCarrier_cat")
      val assemblerArrDelay = new VectorAssembler()
        .setInputCols(featuresNames.toArray)
        .setOutputCol("featuresCorr")
      val scalerArrDelay = new StandardScaler()
        .setInputCol("featuresCorr")
        .setOutputCol("scaledFeaturesCorr")
        .setWithStd(true).setWithMean(true)
      pipeline = new Pipeline()
        .setStages(Array(indexerCity, indexerCarrier, assembler, scaler, assemblerArrDelay, scalerArrDelay))
    }

    val pipelineModel = pipeline.fit(df)
    val dfScaled = pipelineModel.transform(df)

    // Divide data into training and testing according to the options
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

    // Apply regression models
    val resultsLinear = regressionModel("linear", dfTraining, dfTest)
    val resultsTree = regressionModel("decision tree", dfTraining, dfTest)

    printSummary(resultsLinear, resultsTree, dfTraining, featuresNames)
  }
}
