import os
import shutil
import zipfile
import argparse
import sys
from pyspark.sql import SparkSession, functions

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), "tmp")

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
    # return [word for word in tweet if word not in terms_to_remove]
    return words[1:]

def preprocess_dataset(dataset):
    # Run the cleansing UDF for tweet column
    udf_cleansing = functions.udf(cleansing)
    dataset = dataset.withColumn("tweet_cleansed", udf_cleansing(functions.col("tweet")))

    # Tokenizing
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="tweet_cleansed", outputCol="words")
    dataset = tokenizer.transform(dataset)

    # Generating features
    from pyspark.ml.feature import HashingTF
    hashingTF = HashingTF(inputCol="words", outputCol="features")
    dataset = hashingTF.transform(dataset)

    # Generate label indexes
    from pyspark.ml.feature import StringIndexer
    stringIndexer = StringIndexer(inputCol="label", outputCol="labelIndex")
    model = stringIndexer.fit(dataset)
    dataset = model.transform(dataset)

    return dataset

def main():
    parser = argparse.ArgumentParser(description="Train ans save Twitter sentiment analysis classifier model")
    parser.add_argument("--sample_size", type=int, default=None, 
                        help="an integer informing the size of the sample to be taken from training dataset (if not informed, it will use the whole file")
    parser.add_argument("--test_size_frac", type=float, default=.5, 
                        help="an numeric between 0 and 1 informing the fraction of the lines from the sample that will be reserved for testing the dataset (if not informed, it will split the dataset in two)") 
    args = parser.parse_args()
    if "sample_size" in args:
        sample_size = args.sample_size
    else:
        sample_size = None

    if "test_size_frac" in args:
        test_size_frac = args.test_size_frac
    else:
        test_size_frac = .5

    # Unzip file on a temporary folder                                         
    unzip_files()

    # Start Spark session, load the dataset into a Spark DataFrame and then adjust column names
    spark = SparkSession.builder.master("local").appName("Training Twitter Sentiment Analysis").getOrCreate()
    training_data = spark.read.load(
        "tmp/training.1600000.processed.noemoticon.csv",
        format="csv")
    training_data = training_data.withColumnRenamed("_c0", "label") \
        .withColumnRenamed("_c1", "tweet_id") \
        .withColumnRenamed("_c2", "date") \
        .withColumnRenamed("_c3", "query") \
        .withColumnRenamed("_c4", "user") \
        .withColumnRenamed("_c5", "tweet") 

    # Load the amount of lines for training defined on arg sample_size
    training_data = training_data.sample(sample_size / training_data.count())

    # Preprocess dataset
    training_data = training_data.select(functions.col("label"), functions.col("tweet"))
    training_data = preprocess_dataset(training_data)

    # Split dataset into training and test according to test_size_frac arg
    training, test = training_data.randomSplit([1 - test_size_frac, test_size_frac])
    
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
    model_folder = os.path.join(os.getcwd(), 'saved_models')

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        
    model_full_path = os.path.join(model_folder, "twitter_sentiment_spark")
    nbModel.write().overwrite().save(model_full_path)

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