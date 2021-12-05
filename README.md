# Flights_Delay

This app develops a prediction model for the arrival delay of US flights taken from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7flightsinitialization. It is developed with the Spark API for Scala and built as a sbt project. Its execution in the shell works as follows:
```
spark-submit [spark-options] <PATH-TO-JAR-FILE> [options]
```

When running the code, an output.txt file is created with the experiment results (or an error message if the CSV files are not like those in the previous website). Depending on the option you choose in the execution command the file will display different computations. The allowed options are:
- -c: The model is built using all the variables. It only works with a 70-30 percentage split for training and testing, respectively.
- -t <value>: The experiment splits randomly the training and test sets from the original input data sets. <value> denotes the fraction of instances that is destined to the test split, introduced as a float between 0 and 1. It uses a subset of the variables that has been proved to work well.
- -s: Two file choosers are prompted to the user, one for training files and the other for testing files. It uses the same subset of variables as the previous option.
  
These three options are mutually exclusive and the default one (no options provided) is the following:
```
spark-submit [spark-options] <PATH-TO-JAR-FILE> -t 0.3
```
