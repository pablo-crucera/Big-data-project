import spark.implicits._
import org.apache.spark.ml.feature.{UnivariateFeatureSelector, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors


// Create dataframe from CSV file 
// *MODIFY PATH IF NECESSARY*
val df = spark.read.format("csv").option("header", "true").load("/home/julia/Escritorio/Big-data-project/datosbigdata/2008.csv")

// Join all features in a single column in order to perform Univariate Filter FFS
val assembler = new VectorAssembler()
    .setInputCols(Array("Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "CRSElapsedTime", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut"))
    .setOutputCol("features")

val assembledDF = assembler.transform(df)

// Perform the Univariate Filter FFS
val selector = new UnivariateFeatureSelector()
    .setFeatureType("continuous")
    .setLabelType("continuous")
    .setSelectionMode("numTopFeatures")
    .setSelectionThreshold(1)
    .setFeaturesCol("features")
    .setLabelCol("ArrDelay")
    .setOutputCol("selectedFeatures")

val selectedFeatures = selector
    .fit(assembledDF)
    .transform(assembledDF)
    .select("selectedFeatures")

// Show results
selectedFeatures.show()