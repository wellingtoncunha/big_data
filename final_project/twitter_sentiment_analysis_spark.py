import os
import shutil
import zipfile
import argparse
import sys
import yaml
import pymongo
from pyspark.sql import SparkSession, functions, Row, types
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
parameters = yaml.load(open(parameters))
probability_threshold = parameters["classifying"]["probability_threshold"]
test_execution = parameters["classifying"]["test_execution"]
files_source = parameters["classifying"]["files_source"]
mongo_connection_string = (
    "mongodb://" + parameters["mongodb"]["user"] + 
    ":" + parameters["mongodb"]["password"] + 
    "@" + parameters["mongodb"]["host"] + 
    ":" + str(parameters["mongodb"]["port"]) +
    "/" 
)

if test_execution == False:
    conf = SparkConf()\
        .set("spark.jars", "mongo-spark-connector_2.12-3.0.1.jar") \
        .set("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .set("spark.mongodb.input.uri", mongo_connection_string) \
        .set("spark.mongodb.output.uri", mongo_connection_string)
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 60)

def store_into_database(tweet_id, date, query, user, tweet, polarity, probability):
    db_name = mongo_client["twitter_analysis"]
    db_collection = db_name["sentiment_analysis"]

    #drop if exists based on ID
    rows = db_collection.delete_many({"_id": tweet})
    print(rows.deleted_count, "deleted")
    #insert record
    row = {
        "_id": tweet_id,
        "date": date,
        "query": query,
        "user": user,
        "tweet": tweet,
        "predicted": polarity,
        "probability": probability
    }     
    rows = db_collection.insert_one(row)   
    print(rows.inserted_id, "inserted")
    return "Stored"

def get_probability(probability_vector, predicted_label_index):
    probability_array = probability_vector.tolist()
    return probability_array[int(predicted_label_index)]

def load_test_dataset(spark):
# Unzip and load test dataset
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)
        
    if not os.path.exists(temporary_folder):
        os.makedirs(temporary_folder)
        
    local_file_name = os.path.join(base_folder, "training_dataset", "trainingandtestdata.zip")
    with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
        zip_ref.extractall(temporary_folder)
    
    if files_source == "hdfs":
        test_dataset = spark.read.load(
            "/tmp/testdata.manual.2009.06.14.csv",
            format="csv")
    elif:
        test_dataset = spark.read.load(
            "tmp/testdata.manual.2009.06.14.csv",
            format="csv")

    test_dataset = test_dataset.withColumnRenamed("_c0", "label") \
        .withColumnRenamed("_c1", "tweet_id") \
        .withColumnRenamed("_c2", "date") \
        .withColumnRenamed("_c3", "query") \
        .withColumnRenamed("_c4", "user") \
        .withColumnRenamed("_c5", "tweet") 

    return test_dataset

def cleansing(tweet):
# Cleansing tweet
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords 
    from string import punctuation 
    from bs4 import BeautifulSoup
    import re
    
    terms_to_remove = set(stopwords.words("english") + ["USERTAGGING","URL"])
    tweet = BeautifulSoup(tweet, 'html.parser').get_text() # Extracts text from HTML (just in case!)
    tweet = tweet.lower() # Converts text to lower-case
    tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", "URL", tweet) # Replces URLs by URL constan
    tweet = re.sub("@[^\s]+", "USERTAGGING", tweet) # Replaces usernames by USERTAGGING constant 
    tweet = re.sub(r"#([^\s]+)", r"\1", tweet) # Removes the # in #hashtag
    for p in punctuation: 
        tweet = tweet.replace(p, "") # Removes punctiation
    tweet = word_tokenize(tweet) # Creates a list of words
    words = ""
    for each_word in tweet:
        if each_word not in terms_to_remove:
            words = words + " " + each_word

    return words[1:]

def classify_tweets(inbound_dataset):
    # Run the cleansing UDF for tweet column
    udf_cleansing = functions.udf(cleansing)
    inbound_dataset = inbound_dataset.withColumn("tweet_cleansed", udf_cleansing(functions.col("tweet")))

    # Tokenizing
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="tweet_cleansed", outputCol="words")
    inbound_dataset = tokenizer.transform(inbound_dataset)

    # Generating features
    from pyspark.ml.feature import HashingTF
    features_generator = HashingTF(inputCol="words", outputCol="features")
    inbound_dataset = features_generator.transform(inbound_dataset)

    # # Generate label indexes
    # from pyspark.ml.feature import StringIndexer
    # string_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
    # model = string_indexer.fit(inbound_dataset)
    # inbound_dataset = model.transform(inbound_dataset)

    model_folder = os.path.join(os.getcwd(), "saved_models")
    model_full_path = os.path.join(model_folder, "twitter_sentiment_spark")
    if not os.path.exists(model_folder):
        print("model does not exists")

    from pyspark.ml.classification import NaiveBayesModel
    loaded_model = NaiveBayesModel.load(model_full_path)

    # Classifying using saved model
    classified = loaded_model.transform(inbound_dataset)

    spark = getSparkSessionInstance(inbound_dataset.rdd.context.getConf())
    labels = spark.read.load(
        os.path.join(model_folder, "labels.csv"),
        format="csv", header=True)

    classified = classified.join(labels, classified["NB_pred"] == labels["label_id"])

    udf_get_probability = functions.udf(get_probability)
    classified = classified.withColumn("probability", udf_get_probability(
        functions.col("NB_prob"), functions.col("NB_pred")))

    classified = classified.withColumn(
        "label_predicted", 
        functions.when(classified.probability < probability_threshold, "2")
        .otherwise(classified.label_predicted))
        
    return classified

def getSparkSessionInstance(sparkConf):
    if ("sparkSessionSingletonInstance" not in globals()):
        globals()["sparkSessionSingletonInstance"] = SparkSession \
            .builder \
            .config(conf=sparkConf) \
            .getOrCreate()
    return globals()["sparkSessionSingletonInstance"]


def process_streaming(time, rdd):
    if rdd.count() > 0:
        spark = getSparkSessionInstance(rdd.context.getConf())
        
        rdd_rows = rdd.map(lambda w: Row(full_tweet = w))
    
        inbound_dataset = spark.createDataFrame(rdd_rows)
        # inbound_dataset = tweet_stream.select
        inbound_dataset = inbound_dataset.select(functions.split(functions.col("full_tweet"), "\t").alias("full_tweet_array")).drop("full_tweet")
        inbound_dataset = inbound_dataset.withColumn("tweet_id", functions.col("full_tweet_array")[0])
        inbound_dataset = inbound_dataset.withColumn("date", functions.col("full_tweet_array")[1])
        inbound_dataset = inbound_dataset.withColumn("query", functions.col("full_tweet_array")[2])
        inbound_dataset = inbound_dataset.withColumn("user", functions.col("full_tweet_array")[3])
        inbound_dataset = inbound_dataset.withColumn("tweet", functions.col("full_tweet_array")[4])

        outbound_dataset = classify_tweets(inbound_dataset)
        outbound_dataset.describe()
        mongo_items = outbound_dataset.select(
            functions.col("tweet_id").cast("string").alias("_id"), 
            functions.col("date"), 
            functions.col("query").cast("string"), 
            functions.col("user").cast("string"), 
            functions.col("tweet").cast("string"), 
            functions.col("label_predicted").cast("string"),
            functions.col("probability").cast("float")
        )
        
        mongo_items.write.format("mongo").mode("append").option("database",
            "twitter_analysis").option("collection", "sentiment_analysis").save()

        #outbound_dataset.select("tweet_id", "date", "query", "user", "tweet", "label_predicted", "probability").show()
# https://spark.apache.org/docs/latest/streaming-programming-guide.html#dataframe-and-sql-operations
# https://docs.cloudera.com/documentation/enterprise/latest/topics/cdh_ig_running_spark_on_yarn.html
# https://sparkbyexamples.com/spark/spark-submit-command/


def main():
    if test_execution == True:
        # Start Spark session, load the dataset into a Spark DataFrame and then adjust column names
                    # 
        spark = SparkSession \
            .builder \
            .appName("Twitter Sentiment Analysis") \
            .config("spark.jars", "mongo-spark-connector_2.12-3.0.1.jar") \
            .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")\
            .config("spark.mongodb.input.uri", mongo_connection_string) \
            .config("spark.mongodb.output.uri", mongo_connection_string) \
            .getOrCreate()
        # spark = SparkSession.builder.master("local").appName("Training Twitter Sentiment Analysis").getOrCreate()
        inbound_dataset = load_test_dataset(spark)
        
        outbound_dataset = classify_tweets(inbound_dataset)

        # Saving evaluation with test dataset
        # It is important to note that our training set didn't have any Neutral (polarity = 2) single case
        test_folder = os.path.join(os.getcwd(), 'test_model')    
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)

        total = outbound_dataset.count()
        correct = outbound_dataset.where(outbound_dataset['label'] == outbound_dataset['label_predicted']).count()
        accuracy = correct/total
        sys.stdout = open(os.path.join(test_folder, "evaluation.txt"), "w")
        print(
            "\nTotal:", total, 
            "\nCorrect:", correct, 
            "\nAccuracy:", accuracy)
        sys.stdout.close()

        # Save Dataset
        file_name = os.path.join(test_folder, "outbound_test.csv")
        outbound_dataset = outbound_dataset.select("label", "tweet_id", "date", "user", "tweet", "label_predicted", "probability")
        outbound_dataset.toPandas().to_csv(file_name, index=False)

        outbound_dataset = outbound_dataset.select(
            functions.col("tweet_id").cast("string").alias("_id"), 
            functions.col("date"), 
            #functions.col("query").cast("string"), 
            functions.col("user").cast("string"), 
            functions.col("tweet").cast("string"), 
            functions.col("label_predicted").cast("string"),
            functions.col("probability").cast("float")
        )
        
        outbound_dataset.write.format("mongo").mode("append").option("database",
            "twitter_analysis").option("collection", "sentiment_analysis_test").save()

    else:
        os.system("clear")
        # conf = SparkConf().set("spark.jars", "mongo-spark-connector_2.12-3.0.1.jar")

        TCP_IP = parameters["spark"]["host"]
        TCP_PORT = parameters["spark"]["port"]
        tweet_stream = ssc.socketTextStream(TCP_IP, TCP_PORT)

        ##tweet_stream = tweet_stream.flatMap(lambda line: line.split("\t"))
        tweet_stream.foreachRDD(process_streaming)

        # start the streaming computation
        ssc.start()

        # wait for the streaming to finish
        ssc.awaitTermination()


    # Delete temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)    

if __name__ == "__main__":
    main()