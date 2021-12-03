name := "Flights_Delay"
version := "1.0.0"
scalaVersion := "2.12.15"
libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "3.2.0",
"org.apache.spark" %% "spark-sql" % "3.2.0",
"org.apache.spark" %% "spark-mllib" % "3.2.0")
Compile/mainClass := Some("flights.FlightsDelayApp")
