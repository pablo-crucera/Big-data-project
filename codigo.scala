// IMPORT AND LOAD DATASET

val df = spark.read.format("csv").option("header", "true").load("/home/julia/Escritorio/Big-data-project/datosbigdata/2008.csv")


// PREPROCESS DATA

import spark.implicits._
import org.apache.spark.sql.functions.{col, column, expr}
val r=df.select("Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "UniqueCarrier", "FlightNum", "TailNum", "CRSElapsedTime", "ArrDelay", "DepDelay", "Origin", "Dest", "Distance", "TaxiOut", "Cancelled", "CancellationCode")


