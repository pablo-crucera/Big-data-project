import org.apache.spark.sql.SparkSession
import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{PCA, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.stat._
import org.apache.spark.sql._
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.expressions.UserDefinedFunction
import javax.swing.filechooser.FileNameExtensionFilter
import java.io.{File, PrintWriter}
import javax.swing.JFileChooser
import scala.annotation.tailrec


class SparkPredictor {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  // Transform variables with format hhmm to minutes after 00:00
  val parseTime: UserDefinedFunction = udf((s: Int) => s % 100 + (s / 100) * 60)
  var checking = false
  var trainFiles = Array.empty[File]
  var testFiles = Array.empty[File]
  var testPercentage = 0.0
  val outFile = new File("output.txt")
  val output = new PrintWriter(outFile)

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
   * Adds to the DataFrame a  new column that represents de Mahalanobis distance to the mean of each row
   * @param df a DataFrame
   * @param inputCol name of the column where the features are stored
   * @param k number of features stored in the column inputCol
   * @return
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

  def closeOutput(): Unit = {
    output.close()
  }

  /**
   * @param testing a boolean to know if the files to load are for testing
   * @param checking a boolean to know if the data to load is for previous checks
   * @return a DataFrame obtained after reading and transforming the file
   */
  def loadData(testing: Boolean, checking: Boolean): DataFrame = {
    var files = trainFiles
    if (testing)
      files = testFiles

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
  }

  /**
   * Creates and applies a regression model with the training and testing DataFrames received
   * @param regType type of regression
   * @param dfTrain training DataFrame
   * @param dfTest testing DataFrame
   * @return a triple with root mean squared error, R square and adjusted R square
   */
  def regressionModel(regType: String, dfTrain: DataFrame, dfTest: DataFrame): (Double, Double, Double) = {
    var pos = 0
    val regTypes = Array(new LinearRegression()
                            .setFeaturesCol("scaledFeatures")
                            .setLabelCol("ArrDelay")
                            .setMaxIter(10)
                            .setElasticNetParam(0.8),
                          new DecisionTreeRegressor()
                            .setFeaturesCol("scaledFeatures")
                            .setLabelCol("ArrDelay"))
    if(regType == "decision tree")
      pos = 1

    val pipeline = new Pipeline().setStages(Array(regTypes(pos)))
    val model = pipeline.fit(dfTrain).transform(dfTest)
    val predictions = model.select("prediction").rdd.map(_.getDouble(0))
    val labels = model.select("ArrDelay").rdd.map(_.getDouble(0))
    val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    val Rsquare = new RegressionMetrics(predictions.zip(labels)).r2
    val n = dfTrain.count
    model.select("ArrDelay", "prediction").show()
    val k = 7
    val Rsquare_adjusted = 1 - ((1 - Rsquare) * (n - 1)) / (n - k - 1)

    (RMSE, Rsquare, Rsquare_adjusted)
  }

  /**
   * Executes the predictor according to the values of the attributes
   */
  def run(): Unit = {
    val df = loadData(false, checking)
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

    if (checking) {
      for (c <- df.columns)
        output.printf("Column %s: %s null values\n", c, df.filter(col(c).isNull || col(c) === "NA").count().toString)

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

    testPercentage match {
      case 0 =>
        dfTest = pipelineModel.transform(loadData(true, checking))
      case -1 => // Cross-validation
      case _ =>
    }

    // Perform Principal Component Analysis and delete outliers from the trainig set
    val K = 3
    val dfPCA = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pca-features")
      .setK(K).fit(training)
      .transform(training)

    val dfMahalanobis: DataFrame = addMahalanobis(dfPCA, "pca-features", K)
    val quantiles = dfMahalanobis.stat.approxQuantile("mahalanobis", Array(0.25, 0.75), 0.0)
    val Q1 = quantiles(0)
    val Q3 = quantiles(1)
    val IQR = Q3 - Q1
    val upperRange = Q3 + 1.5 * IQR
    val dfTraining = dfMahalanobis.filter($"mahalanobis" < upperRange)

    if (checking) {
      val Row(coeff1: Matrix) = Correlation.corr(dfTraining, "scaledFeaturesCorr").head
      output.println(s"Pearson correlation matrix:\n " + coeff1.toString(16, 100000))
    }

    val regressors = Array("linear", "decision tree")
    for (reg <- regressors) {
      val (rmse, rSquare, rSquareAdj) = regressionModel(reg, dfTraining, dfTest)
      output.println("Root mean squared error (RMSE) for" + reg + s" regression: $rmse")
      output.println("R-square for " + reg + s" regression: $rSquare")
      output.println("R-square adjusted for " + reg + s" regression: $rSquareAdj")
    }
    closeOutput()
  }
}

object MyApp {
  /** Checks if the file has the appropriate structure
   * @param path path of the file to check
   * @return true if the file is ok, false in other case
   */
  def checkFile(path: String): Boolean = {
    true
  }

  /**
   * Shows a file chooser that allows the user to select multiple CSV files
   * @param s a string that tells the user has to select
   * @return the option chosen in the dialog and and array with the paths to the files selected
   */
  def selectFiles(s: String): (Int, Array[File]) = {
    var filePaths = Array.empty[File]
    val chooser = new JFileChooser()
    val extension_filter = new FileNameExtensionFilter("CSV files", "csv") // Show only csv files
    chooser.setCurrentDirectory(new java.io.File("."))
    chooser.setDialogTitle("Select CSV files" + s)
    chooser.setFileFilter(extension_filter)
    chooser.setMultiSelectionEnabled(true); // Allow multiple files selection
    val option = chooser.showOpenDialog(null)
    if (option == JFileChooser.APPROVE_OPTION)
      filePaths = chooser.getSelectedFiles
    (option, filePaths)
  }

  /**
   * Parses the command line arguments
   * @param args list of the command line arguments of the program
   * @param check default value for check
   * @param perc default value for perc
   * @return a triple with the values for check, testing percentage and different files option
   */
  @tailrec
  def parseArgs(args: List[String], check: Boolean, perc: Double): (Boolean, Double) = args match {
    case Nil => (check, perc)
    case "-t" :: num :: rest => parseArgs(rest, check, num.toDouble)
    case "-c" :: rest => parseArgs(rest, true, perc)
    case "-s" :: rest => parseArgs(rest, check, 0)
    case "--cv" :: rest => parseArgs(rest, check, -1)
  }

  def main(args: Array[String]): Unit = {
    val usage =
      """
        |-t <value>   percentage used for testing
        |-c           check other configurations
        |-s           select different files for training and testing
        |--cv         perform cross-validation
        |""".stripMargin
    val argsList = args.toList
    val tupleArgs = parseArgs(argsList, false, 0.3)

    val predictor = new SparkPredictor()
    predictor.testPercentage = tupleArgs._2
    predictor.checking = tupleArgs._1

    var tuple1 = (JFileChooser.APPROVE_OPTION, Array.empty[File])
    var tuple2 = (JFileChooser.APPROVE_OPTION, Array.empty[File])

    while (tuple1._1 == JFileChooser.APPROVE_OPTION && tuple2._1 == JFileChooser.APPROVE_OPTION &&
          (tuple1._2.isEmpty || (tuple2._2.isEmpty && predictor.testPercentage == 0))) {
      if (predictor.testPercentage > 0)
        tuple1 = selectFiles("")
      else {
        tuple1 = selectFiles("for training")
        tuple2 = selectFiles("for testing")
      }
    }

    if (tuple1._1 == JFileChooser.APPROVE_OPTION && tuple2._1 == JFileChooser.APPROVE_OPTION && !tuple1._2.isEmpty &&
        (!tuple2._2.isEmpty || predictor.testPercentage != 0)) {
      predictor.trainFiles = tuple1._2
      predictor.testFiles = tuple2._2
      predictor.run()
    }
  }
}
