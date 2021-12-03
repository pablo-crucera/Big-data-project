package flights

import java.io.File
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter

object FlightsDelayApp {
  /**
   * Shows a file chooser that allows the user to select multiple CSV files
   *
   * @param s a string that tells the user has to select
   * @return the option chosen in the dialog and and array with the paths to the files selected
   */
  def selectFiles(s: String): (Boolean, Array[File]) = {
    var filePaths = Array.empty[File]
    val chooser = new JFileChooser()
    val extension_filter = new FileNameExtensionFilter("CSV files", "csv") // Show only csv files
    chooser.setCurrentDirectory(new java.io.File("."))
    chooser.setDialogTitle("Select CSV files" + s)
    chooser.setFileFilter(extension_filter)
    chooser.setMultiSelectionEnabled(true); // Allow multiple files selection
    val option = chooser.showOpenDialog(null)
    val selected = chooser.getSelectedFiles
    if (option == JFileChooser.APPROVE_OPTION && selected.map(f => f.exists()).reduce(_&&_))
      filePaths = selected
    (option == JFileChooser.APPROVE_OPTION, filePaths)
  }

  /**
   * Parses the command line arguments
   *
   * @param args  list of the command line arguments of the program
   * @param check default value for check
   * @param perc  default value for perc
   * @return a triple with three values: parsing error, checking selected and testing percentage selected
   */
  def parseArgs(args: List[String], check: Boolean, perc: Double): (Boolean, Boolean, Double) = args match {
    case "-t" :: num :: _ =>
      try {
        val newPerc = num.toDouble
        (newPerc < 0 || newPerc > 1, check, newPerc)
      } catch {
        case _: NumberFormatException => (true, check, perc)
      }
    case "-c" :: rest => (rest.nonEmpty, true, perc)
    case "-s" :: rest => (rest.nonEmpty, check, 0)
    case _ => (false, check, perc)
  }

  def errorMessage(args: Array[String]): Unit = {
    val usage =
      """
        |Usage: spark-submit [spark-submit-options] <app-jar> [-t <value> | -c | -s]
        |
        |Options:
        | -t <value>   percentage used for testing
        | -c           check other configurations
        | -s           select different files for training and testing
        |""".stripMargin
    println("ERROR: Unrecognized option: " + args.mkString(" "))
    println(usage)
  }

  def main(args: Array[String]): Unit = {
    if (args.length > 2) {
      errorMessage(args)
      sys.exit(1)
    }

    val argsList = args.toList
    val tupleArgs = parseArgs(argsList, false, 0.3)

    if (tupleArgs._1) {
      errorMessage(args)
      sys.exit(1)
    }
    val predictor = new SparkPredictor()
    predictor.testPercentage = tupleArgs._3
    predictor.checking = tupleArgs._2

    var tuple1 = (true, Array.empty[File])
    var tuple2 = (true, Array.empty[File])

    while (tuple1._1 && tuple2._1 && (tuple1._2.isEmpty || (tuple2._2.isEmpty && predictor.testPercentage == 0))) {
      if (predictor.testPercentage > 0)
        tuple1 = selectFiles("")
      else {
        tuple1 = selectFiles(" for training")
        tuple2 = selectFiles(" for testing")
      }
    }

    if (tuple1._1 && tuple2._1 && !tuple1._2.isEmpty && (!tuple2._2.isEmpty || predictor.testPercentage != 0)) {
      predictor.trainFiles = tuple1._2
      predictor.testFiles = tuple2._2
      predictor.run()
    }
  }
}
