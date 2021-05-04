import os
import shutil
import zipfile
import argparse
import sys
import yaml
from pyspark.sql import SparkSession, functions, Row
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
parameters = yaml.load(open(parameters))

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



def classify_tweets(inbound_dataset, probability_threshold):
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

    # Generate label indexes
    from pyspark.ml.feature import StringIndexer
    string_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
    model = string_indexer.fit(inbound_dataset)
    inbound_dataset = model.transform(inbound_dataset)

    model_folder = os.path.join(os.getcwd(), "saved_models")
    model_full_path = os.path.join(model_folder, "twitter_sentiment_spark")
    if not os.path.exists(model_folder):
        print("model does not exists")

    from pyspark.ml.classification import NaiveBayesModel
    loaded_model = NaiveBayesModel.load(model_full_path)

    # Classifying using saved model
    classified = loaded_model.transform(inbound_dataset)

    labels = classified.select("labelIndex", "label").distinct() \
        .withColumnRenamed("label", "label_predicted") \
        .withColumnRenamed("labelIndex", "label_id")

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
        
        rowRdd = rdd.map(lambda w: Row(word=w))
        wordsDataFrame = spark.createDataFrame(rowRdd)

        wordsDataFrame.show()



def main():
    parser = argparse.ArgumentParser(description="Twitter sentiment analysis classification")
    parser.add_argument("--probability_threshold", type=float, default=None, 
        help="an numeric between 0.5 and 1 that will be used as a threshold to classify the tweet. If probability is lower than it, then the twitter is classified as neutral (polarity=2)")
    parser.add_argument("--test", dest="test", action="store_true",
        help="run the classification for the test file available and save it to /test folder")
    parser.add_argument("--search_keyword", type=str, 
        help="a word used to search Twitter")
    parser.add_argument("--fetch_size", type=int, default=100, 
        help="an integer with the amount of tweets to fetch during each run (default is 100)")


    args = parser.parse_args()
    if "probability_threshold" in args:
        probability_threshold = args.probability_threshold
    else:
        probability_threshold = 0

    if "test" in args:
        if args.test == True:
            # Start Spark session, load the dataset into a Spark DataFrame and then adjust column names
            spark = SparkSession.builder.master("local").appName("Training Twitter Sentiment Analysis").getOrCreate()
            inbound_dataset = load_test_dataset(spark)
            
            outbound_dataset = classify_tweets(inbound_dataset, probability_threshold)

            
            # Saving evaluation with test dataset
            # It is important to note that our training set didn't have any Neutral (polarity = 2) single case

            test_folder = os.path.join(os.getcwd(), 'test_model')    
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            total = outbound_dataset.count()
            correct = outbound_dataset.where(outbound_dataset['labelIndex'] == outbound_dataset['NB_pred']).count()
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
        else:
            sc = SparkContext()
            ssc = StreamingContext(sc, 60)

            TCP_IP = parameters["spark"]["host"]
            TCP_PORT = parameters["spark"]["port"]
            dataStream = ssc.socketTextStream(TCP_IP, TCP_PORT)

            dataStream.foreachRDD(process_streaming)

            # start the streaming computation
            ssc.start()

            # wait for the streaming to finish
            ssc.awaitTermination()


    # Delete temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)    

if __name__ == "__main__":
    main()