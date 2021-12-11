# Flights Delay

## Description

Project developed for the Big Data course of the Master's Programme in Data Science of UPM.

This app develops a prediction model for the arrival delay of US flights (taken from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HG7NV7flightsinitialization)) using [Apache Spark](https://spark.apache.org/). A more detailed description of its behavior is shown in <a href="./project.pdf" target="_blank">project.pdf</a> and <a href="./report.pdf" target="_blank">report.pdf</a>.

## Requirements
The project has been developed with the following sotware:
- Apache Spark 3.2.0
- Scala 2.12.15
- SBT 1.5.5
- Java 8 JDK or Java 11 JDK

## Installation
Compile and package using the following command:
```
sbt package
```
After this, a `flights_delay_2.12-1.0.0.jar` file will be created under a `target/scala-2.12/` directory.

## Usage

The execution of the program in the shell works as follows:
```
spark-submit [spark-options] target/scala-2.12/flights_delay_2.12-1.0.0.jar [options]
```

The allowed options represent different computations and are:
- `-c`: the model is built using all the variables. Input files are selected with a file chooser and a 70-30 percentage split is performed for training and testing, respectively.
- `-t <value>`: the experiment splits randomly the training and test sets from the original input data sets (selected in the same way as previous option describes). `<value>` denotes the fraction of instances that is destined to the test split, introduced as a float between 0 and 1. It uses a subset of the variables that has been proved to work well.
- `-s`: two file choosers are prompted to the user, one for training files and the other for testing files. It uses the same subset of variables as the previous option.

These three options are mutually exclusive and the default one (no options provided) is the following:
```
spark-submit [spark-options] target/scala-2.12/flights_delay_2.12-1.0.0.jar -t 0.3
```

After running the code, an `output.txt` file is created with the experiment results (or an error message if the CSV files are not like those in the previous website).

## Authors
- Pablo Crucera Barrero ([@pablo-crucera](https://github.com/pablo-crucera))
- Júlia Sánchez Martínez ([@Julia-upc](https://github.com/Julia-upc))
- Javier Gallego Gutiérrez ([@javiegal](https://github.com/javiegal))
