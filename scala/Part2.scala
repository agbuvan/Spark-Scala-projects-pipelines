import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import scala.collection.mutable

object Part2 {
  val sc = new SparkContext(new SparkConf().setAppName("Part2"))
  var count = 1
  var rdd :RDD[String]= null
  var bestAirline = ""
  var worstAirline = ""
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("Simple Application").getOrCreate()
    val sqlContext = spark.sqlContext
    import spark.implicits._
    var train = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(args(0))
    train = train.filter($"text".isNotNull)

    def airlineSentiment() = udf[Double, String] { a => val x = a match {
      case "positive" => 5.0;
      case "neutral" => 2.5;
      case "negative" => 1.0;
    }; x;
    }

    val train_df = train.withColumn("airline_sentiment", airlineSentiment()($"airline_sentiment"))
    val average = train_df.groupBy("airline").agg(avg("airline_sentiment").as("airline_sentiment")).sort(desc("airline_sentiment"))
    average.show()
    val top = average.agg(max("airline_sentiment")).head().getDouble(0)
    val least = average.agg(min("airline_sentiment")).head().getDouble(0)
    bestAirline = average.filter($"airline_sentiment" === top).select($"airline").head().getString(0)
    worstAirline = average.filter($"airline_sentiment" === least).select($"airline").head().getString(0)
    val filtered_best = train_df.filter($"airline" === bestAirline).select("text")
    val filtered_worst = train_df.filter($"airline" === worstAirline).select("text")
    topic_Model(filtered_best)
    topic_Model(filtered_worst)
    rdd.coalesce(1,true).saveAsTextFile(args(1))
  }

  def topic_Model(filteredMax: DataFrame) : Unit ={
    val rdd_filteredMax =
      filteredMax.rdd
        .map(row => {
          val text = row.getString(0)

          (text)
        })
    val stopwords = StopWordsRemover.loadDefaultStopWords("english").toSet
    val tokenized: RDD[Seq[String]] =
      rdd_filteredMax.map(_.toLowerCase.split("\\s")).map(_.filter(_.length > 3).filter(token =>     !stopwords.contains(token)).filter(_.forall(java.lang.Character.isLetter)))
    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)
    val numStopwords = 20
    val vocabArray: Array[String] =
      termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }
    val numTopics = 5
    val lda = new LDA().setK(numTopics).setMaxIterations(5)
    val ldaModel = lda.run(documents)
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    var output = ""
    output += "The Best Airline is :"+ bestAirline + " and the worst airline is :" + worstAirline +"\n"
    topicIndices.foreach { case (terms, termWeights) =>
      output += "Topic:" + "\n"
      terms.zip(termWeights).foreach { case (term, weight) =>
        output += {vocabArray(term.toInt)}
        output += "\t" + weight + "\n"
      }
      output += "\n\n"

    }
    if(count == 1){
      rdd = sc.parallelize(List(output))
    }else{
      val rdd1 = sc.parallelize(List(output))
      rdd = rdd ++ rdd1
    }
    count = count+1
  }
}