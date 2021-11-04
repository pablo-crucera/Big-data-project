import spark.implicits._
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{StringIndexer, StandardScaler, VectorAssembler, UnivariateFeatureSelector}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors

// Set schema of the dataframe
val schema = StructType(Array(
    StructField("Year",IntegerType,true),
    StructField("Month",IntegerType,true),
    StructField("DayofMonth",IntegerType,true),
    StructField("DayOfWeek", IntegerType, true),
    StructField("DepTime", StringType, true),
    StructField("CRSDepTime", StringType, true),
    StructField("ArrTime", StringType, true),
    StructField("CRSArrTime", StringType, true),
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

// Load data and drop unuseful variables and rows with a null value for the target variable
val filePath = "/home/javier/Documents/master/bddv/data/2002.csv"
val df = spark.read.option("header", "true")
  .schema(schema)
  .csv(filePath)
  .drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay", "Cancelled", "CancellationCode", "TailNum")
  .filter(col("ArrDelay").isNotNull)

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
val pipeline = new Pipeline()
  .setStages(Array(indexer_city, indexer_carrier))
val df2 = pipeline.fit(df)
  .transform(df)
  .drop("Origin", "Dest", "UniqueCarrier")

// Transform variables with format hhmm to minutes after 00:00
val parseTime = udf((s: String) => if (s.length < 3) {s.toInt} else {s.dropRight(2).toInt * 60 + s.takeRight(2).toInt})
val df3 = df2.withColumn("DepTime",parseTime($"DepTime"))
  .withColumn("CRSDepTime",parseTime($"CRSDepTime"))
  .withColumn("CRSArrTime",parseTime($"CRSArrTime"))

val features_names = df3.columns.filter(! _.contains("ArrDelay"))
val assembler = new VectorAssembler()
  .setInputCols(features_names)
  .setOutputCol("features")

val output = assembler.transform(df3).select("features", "ArrDelay")

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithStd(true)
  .setWithMean(true)
val scaledData = scaler.fit(output)
  .transform(output)
  .select("scaledFeatures", "ArrDelay")

// Perform the Univariate Filter FFS
val selector = new UnivariateFeatureSelector()
  .setFeatureType("continuous")
  .setLabelType("continuous")
  .setSelectionMode("numTopFeatures")
  .setSelectionThreshold(1)
  .setFeaturesCol("scaledFeatures")
  .setLabelCol("ArrDelay")
  .setOutputCol("selectedFeatures")

val selectedFeatures = selector.fit(scaledData)
  .transform(scaledData)
  .select("selectedFeatures")

// Show results
selectedFeatures.show()
