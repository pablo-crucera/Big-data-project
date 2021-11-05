import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, StandardScaler, VectorAssembler, UnivariateFeatureSelector}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors

object MyApp{
  def main(args: Array[String]) {
    val spark = SparkSession.builder().getOrCreate()

    import spark.implicits._

    // Set schema of the dataframe
    val schema = StructType(Array(
        StructField("Year",IntegerType,true),
        StructField("Month",IntegerType,true),
        StructField("DayofMonth",IntegerType,true),
        StructField("DayOfWeek", IntegerType, true),
        StructField("DepTime", IntegerType, true),
        StructField("CRSDepTime", IntegerType, true),
        StructField("ArrTime", StringType, true),
        StructField("CRSArrTime", IntegerType, true),
        StructField("UniqueCarrier", StringType, true),
        StructField("FlightNum", IntegerType, true),
        StructField("TailNum", IntegerType, true),
        StructField("ActualElapsedTime", StringType, true),
        StructField("CRSElapsedTime", IntegerType, true),
        StructField("AirTime", StringType, true),
        StructField("ArrDelay", IntegerType, true),
        StructField("DepDelay", IntegerType, true),
        StructField("Origin", StringType, true),
        StructField("Dest", StringType, true),
        StructField("Distance", IntegerType, true),
        StructField("TaxiIn", StringType, true),
        StructField("TaxiOut", IntegerType, true),
        StructField("Cancelled", IntegerType, true),
        StructField("CancellationCode", StringType, true),
        StructField("Diverted", StringType, true),
        StructField("CarrierDelay", StringType, true),
        StructField("WeatherDelay", StringType, true),
        StructField("NASDelay", StringType, true),
        StructField("SecurityDelay", StringType, true),
        StructField("LateAircraftDelay", StringType, true)
      ))

    // Transform variables with format hhmm to minutes after 00:00
    val parseTime = udf((s: Int) => s % 100 + (s / 100) * 60)


    // Load data, drop unuseful variables and rows with a null value for the target variable and transform time variables
    val filePath = "/home/javier/Documents/master/bddv/data/2002.csv"
    val df = spark.read.option("header", "true")
      .schema(schema)
      .csv(filePath)
      .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "Cancelled", "CancellationCode", "TailNum")
      .filter(col("ArrDelay").isNotNull)
      .withColumn("DepTime",parseTime($"DepTime"))
      .withColumn("CRSDepTime",parseTime($"CRSDepTime"))
      .withColumn("CRSArrTime",parseTime($"CRSArrTime"))

    // Check null values of each variable
    for (c <- df.columns){
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
      .setWithStd(true)
      .setWithMean(true)

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
      .setStages(Array(indexer_city, indexer_carrier, assembler, scaler, selector))
    val dfTransformed = pipeline.fit(df).transform(df)

    // Show results
    dfTransformed.show()
  }
}
