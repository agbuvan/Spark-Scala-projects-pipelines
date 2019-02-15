import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.classification.{RandomForestClassifier}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD

object Part1 {
  val sc = new SparkContext(new SparkConf().setAppName("Part1"))
  var rdd :RDD[String]= null

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val sqlContext = spark.sqlContext
    import spark.implicits._
    var training = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(args(0))
    training = training.filter($"text".isNotNull)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val stop = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("stopWords")
    val hashingTF = new HashingTF()
      .setInputCol(stop.getOutputCol)
      .setOutputCol("features")
    val convertedLabel = new StringIndexer()
      .setInputCol("airline_sentiment")
      .setOutputCol("label")
    val lr = new LogisticRegression()
      .setMaxIter(10).setFeaturesCol("features").setLabelCol("label")
    val rf = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label")

    val pipeline_rf = new Pipeline()
      .setStages(Array(tokenizer, stop, hashingTF, convertedLabel, rf))

    val pipeline_lr = new Pipeline()
      .setStages(Array(tokenizer, stop, hashingTF, convertedLabel, lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(rf.maxDepth, Array(20,30))
      .build()

    val paramGrid1 = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val cv_rf = new CrossValidator()
      .setEstimator(pipeline_rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cv_lr = new CrossValidator()
      .setEstimator(pipeline_lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid1)
      .setNumFolds(3)

    val Array(train, test) = training.randomSplit(Array(0.75,0.25))
    val cvModel_rf = cv_rf.fit(train)
    val cvModel_lr = cv_lr.fit(train)
    val transformModel_rf = cvModel_rf.transform(test)
    val transformModel_lr = cvModel_lr.transform(test)
    var accuracy = 0.0
    var accuracy1 = 0.0
    var precision= 0.0
    var recall= 0.0
    var f1score= 0.0
    var precision1= 0.0
    var recall1= 0.0
    var f1score1= 0.0

    //RF
    def displayMetrics(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      accuracy = metrics.accuracy
      precision = metrics.weightedPrecision
      recall = metrics.weightedRecall
      f1score = metrics.weightedFMeasure
    }

    val rfPredictionAndLabels = transformModel_rf.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    displayMetrics(rfPredictionAndLabels)

    def displayMetrics1(pAndL : RDD[(Double, Double)]) {
      val metrics = new MulticlassMetrics(pAndL)
      accuracy1 = metrics.accuracy
      precision1 = metrics.weightedPrecision
      recall1 = metrics.weightedRecall
      f1score1 = metrics.weightedFMeasure
    }

    val lrPredictionAndLabels = transformModel_lr.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}
    displayMetrics1(lrPredictionAndLabels)

    var output = ""
    output+= "Accuracy for Logistic Regression: \t" + accuracy1 + "\n"
    output+= "Precision for Logistic Regression: \t" + precision1 + "\n"
    output+= "Recall  for Logistic Regression: \t" + recall1 + "\n"
    output+= "F1Score  for Logistic Regression: \t" + f1score1 + "\n"
    output+= "Accuracy for Random Forest: \t" + accuracy + "\n"
    output+= "Precision for Random Forest: \t" + precision + "\n"
    output+= "Recall  for Random Forest: \t" + recall + "\n"
    output+= "F1Score  for Random Forest: \t" + f1score + "\n"
    rdd = sc.parallelize(List(output))
    rdd.coalesce(1,true).saveAsTextFile(args(1))
  }
}
