import os
import shutil
import zipfile
import argparse
import sys
import yaml
from pyspark.sql import SparkSession, functions
import nltk
from nltk.tokenize import word_tokenize
from string import punctuation 
from bs4 import BeautifulSoup
import re
nltk.download('stopwords')
nltk.download('punkt')

# Start Spark session, load the dataset into a Spark DataFrame and then adjust column names
spark = SparkSession.builder.appName("Training Twitter Sentiment Analysis").getOrCreate()

# Load folders
base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), "tmp")

# Load execution parameter (we replaced args because it is easy for calling :) )
parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
parameters = yaml.load(open(parameters))
sample_size = parameters["training"]["sample_size"]
test_size_fraction = parameters["training"]["test_size_fraction"]
files_source = parameters["training"]["files_source"]

def unzip_files():
# Unzip file on a temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)
        
    if not os.path.exists(temporary_folder):
        os.makedirs(temporary_folder)
        
    local_file_name = os.path.join(base_folder, "training_dataset", "trainingandtestdata.zip")
    with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
        zip_ref.extractall(temporary_folder)

def cleansing(tweet):
# Cleansing tweet
    from nltk.corpus import stopwords 
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

def main():
    # Unzip file on a temporary folder                                         
    unzip_files()

    if files_source == "hdfs":
        training_data = spark.read.load(
            "/tmp/training.1600000.processed.noemoticon.csv",
            format="csv")
    elif files_source == "local":
        training_data = spark.read.load(
            "tmp/training.1600000.processed.noemoticon.csv",
            format="csv")        
    training_data = training_data.withColumnRenamed("_c0", "label") \
        .withColumnRenamed("_c1", "tweet_id") \
        .withColumnRenamed("_c2", "date") \
        .withColumnRenamed("_c3", "query") \
        .withColumnRenamed("_c4", "user") \
        .withColumnRenamed("_c5", "tweet") 

    # Load the amount of lines for training defined on arg sample_size. If equals zero, use the whole dataset
    if sample_size > 0: 
        training_data = training_data.sample(sample_size / training_data.count())

    ## Preprocess dataset
    training_data = training_data.select(functions.col("label"), functions.col("tweet"))

    # Run the cleansing UDF for tweet column
    udf_cleansing = functions.udf(cleansing)
    training_data = training_data.withColumn("tweet_cleansed", udf_cleansing(functions.col("tweet")))

    # Tokenizing
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="tweet_cleansed", outputCol="words")
    training_data = tokenizer.transform(training_data)

    # Generating features
    from pyspark.ml.feature import HashingTF
    hashingTF = HashingTF(inputCol="words", outputCol="features")
    training_data = hashingTF.transform(training_data)

    # Generate label indexes
    from pyspark.ml.feature import StringIndexer
    stringIndexer = StringIndexer(inputCol="label", outputCol="labelIndex")
    model = stringIndexer.fit(training_data)
    training_data = model.transform(training_data)
    
    # Split dataset into training and test according to test_size_frac arg
    training, test = training_data.randomSplit([1 - test_size_fraction, test_size_fraction])
    
    # Training the model
    from pyspark.ml.classification import NaiveBayes
    #Naive bayes
    nb = NaiveBayes(featuresCol="features", labelCol="labelIndex", predictionCol="NB_pred",
                    probabilityCol="NB_prob", rawPredictionCol="NB_rawPred")
    nbModel = nb.fit(training)
    cv = nbModel.transform(test)
    total = cv.count()
    correct = cv.where(cv['labelIndex'] == cv['NB_pred']).count()
    accuracy = correct/total

    # Saving trained model for usage in a Pipeline (so you don't need to re-train everytime you need to use it)
    model_folder = os.path.join(base_folder, 'saved_models')
    print(model_folder)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    model_full_path = os.path.join(model_folder, "twitter_sentiment_spark")
    if files_source == "hdfs":
        model_full_path = "file://" + model_full_path
    nbModel.write().overwrite().save(model_full_path)


    # Save Labels reference table    
    labels = cv.select("labelIndex", "label").distinct() \
        .withColumnRenamed("label", "label_predicted") \
        .withColumnRenamed("labelIndex", "label_id")

    labels.toPandas().to_csv(os.path.join(model_folder, "labels.csv"), index=False)

    # Save evaluations
    sys.stdout = open(os.path.join(model_folder, "evaluation.txt"), "w")
    print(
        "\nTotal:", total, 
        "\nCorrect:", correct, 
        "\nAccuracy:", accuracy)
    sys.stdout.close()

    # Delete temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)    

if __name__ == "__main__":
    main()