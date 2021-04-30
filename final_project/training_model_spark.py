from pyspark.sql import SparkSession, functions
from pyspark import SparkConf, SparkContext
# from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils
import os
import zipfile
import shutil
import argparse
from bs4 import BeautifulSoup
import re



# import pandas as pd
# import urllib.request 

# import nltk
# import sys


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import dill as pickle

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), "tmp")

def cleansing_and_tokenizing(tweet):
# Cleansing and tokenizing tweet
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords 
    from string import punctuation 
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

def extract_features(tweet, vocabulary):
# Extract features (tag words using on twitter into a instance of dictionary)
    tweet_words=set(tweet)
    features={}
    for word in vocabulary.toLocalIterator():
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

class TwitterSentimentAnalysis:
# Sentiment analysis class: packs everything related to train and use the model
    def __init__(self):
        self._stopwords = set(stopwords.words("english") + ["USERTAGGING","URL"]);

    def processTweets(self, list_of_tweets):
    # Process a list of tweets
        processedTweets=[]
        for tweet in list_of_tweets:
            processedTweets.append(
                (
                    self.processTweet(tweet["tweet"]),
                    tweet["polarity"]                    
                )
            )
        return processedTweets
    
    def processTweet(self, tweet):
    # Cleansing and tokenizing tweet
        tweet = BeautifulSoup(tweet).get_text() # Extracts text from HTML (just in case!)
        tweet = tweet.lower() # Converts text to lower-case
        tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", "URL", tweet) # Replces URLs by URL constan
        tweet = re.sub("@[^\s]+", "USERTAGGING", tweet) # Replaces usernames by USERTAGGING constant 
        tweet = re.sub(r"#([^\s]+)", r"\1", tweet) # Removes the # in #hashtag
        for p in punctuation: 
            tweet = tweet.replace(p, "") # Removes punctiation
        tweet = word_tokenize(tweet) # Creates a list of words
        return [word for word in tweet if word not in self._stopwords]





    def trainModel(self, training_features):
        # Pre-process training set (cleansing and tokenizing)
        preprocessed_training_data =  self.processTweets(training_features)
        # Build vocabulary
        self.buildVocabulary(preprocessed_training_data)
        # Building the training features
        training_features = nltk.classify.apply_features(self.extract_features,preprocessed_training_data)
        # Train Naive Bayes model
        self.Classifier = nltk.NaiveBayesClassifier.train(training_features)


def unzip_files():
# Unzip file on a temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)
        
    if not os.path.exists(temporary_folder):
        os.makedirs(temporary_folder)
        
    local_file_name = os.path.join(base_folder, "training_dataset", "trainingandtestdata.zip")
    with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
        zip_ref.extractall(temporary_folder)

def load_training_dataset(sample_size = None, test_size_frac = 0.5):
# Load training dataset and split it into training and test
    training_dataset_path = os.path.join(
        temporary_folder, 
        "training.1600000.processed.noemoticon.csv")

    training_dataset = pd.read_csv(
        training_dataset_path, 
        encoding="latin-1", 
        warn_bad_lines=True,
        error_bad_lines=False,
        header=None, 
        names=["polarity", "tweet_id", "date", "query", "user", "tweet"])
    if sample_size != None:
        training_dataset = training_dataset.sample(sample_size)
    
    testing_dataset = training_dataset.sample(frac = test_size_frac)
    training_dataset = training_dataset.drop(testing_dataset.index)
 
    return training_dataset.to_dict("records"), testing_dataset.to_dict("records")

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

    # Setting Spark Context
    # conf = SparkConf().setAppName("WorstMovies").setMaster("local")
    # sc = SparkContext(conf = conf)

    from pyspark.ml.classification import NaiveBayes, NaiveBayesModel

    ## Generate Labeled file
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
    training_data = training_data.sample(sample_size / 1600000)

    training_data = training_data.select(functions.col("label"), functions.col("tweet"))
    training_data = training_data.withColumn("label", functions.col("label").cast("integer"))

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Word2Vec # https://spark.apache.org/docs/latest/ml-features
    from pyspark.ml.classification import NaiveBayes

    udf_cleansing_and_tokenizing = functions.udf(cleansing_and_tokenizing)
    training_data = training_data.withColumn("tweet_cleansed", udf_cleansing_and_tokenizing(functions.col("tweet")))
    #training_data = training_data.withColumn("tokenized", functions.col("tweet_cleansed"))

    tokenizer = Tokenizer(inputCol="tweet_cleansed", outputCol="words")
    training_data = tokenizer.transform(training_data)
    
    word2vec = Word2Vec(inputCol="words", outputCol="vectorized") 
    model = word2vec.fit(training_data)
    training_data = model.transform(training_data)   



    # hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    # training_data = hashingTF.transform(training_data)

    # idf = IDF(inputCol="rawFeatures", outputCol="features")
    # idfModel = idf.fit(training_data)
    # training_data = idfModel.transform(training_data)

    #training, test = training_data.select("label", "features").randomSplit([0.5, 0.5])
    training.show(5)


#    labeled_points = (training.select("label", "features").rdd.map(lambda row: LabeledPoint(row.label, row.features)))
#    labeled_points.take(5)

    model = NaiveBayes(featuresCol="features", labelCol="label")
    model_train = model.fit(training)
    predictions = model_train.transform(test)
    predictions.show(5)

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)

    model = NaiveBayes .train(labeled_points)

    predictionAndLabel = test.withColumn("predicted", model.predict(functions.col("features")))

    predictionAndLabel.take(5)


    training_data = training_data.withColumn("tweet_cleansed", functions.split("tweet_cleansed", ",")) #.select(functions.col("words"))
    # training_data.show(10)

    # Generate vocabulary (save Vocabulary somewhere):
    # https://stackoverflow.com/questions/64744634/pyspark-dataframe-to-extract-each-distinct-word-from-a-column-of-string-and-put

    vocabulary = training_data.withColumn("word", functions.explode("tweet_cleansed")).select(functions.col("word"))
    print("Count:", vocabulary.count())
    vocabulary = vocabulary.distinct()
    print("Distinct count:", vocabulary.count())
    vocabulary.to


    # Generate features:
    # 
    udf_extract_features = functions.udf(extract_features)
    training_data = training_data.withColumn("features", udf_extract_features(functions.col("tweet"), vocabulary))
    training_data.show(5)
    # training_data, testing_data = load_training_dataset(sample_size = sample_size, test_size_frac=test_size_frac)

    # Train dataset
    # https://spark.apache.org/docs/latest/mllib-naive-bayes.html 
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.mllib.regression.LabeledPoint.html
    # https://larry-lu.github.io/data/Naive-Bayes-news-classification/

    # Build TwitterSEntimentAnalysis class
    twitter_sentiment_classifier = TwitterSentimentAnalysis()
    twitter_sentiment_classifier.trainModel(training_data)

    ## Use the test dataset to evaluate the model
    # Use the classifier to predict every tweet from test dataset
    li = []
    threshold = 0  # We can set a threshold base on probabily (must the greater than .5 and less than 1) and, if not meet we classify as 2-Neutral 
    for each_tweet in testing_data:
        words = twitter_sentiment_classifier.processTweet(each_tweet["tweet"])
        features = twitter_sentiment_classifier.extract_features(words)
        predicted = twitter_sentiment_classifier.Classifier.classify(features)
        probability = twitter_sentiment_classifier.Classifier.prob_classify(features).prob(predicted)
        row = {
            "polarity": each_tweet["polarity"],
            "tweet_id": each_tweet["tweet_id"],
            "date": each_tweet["date"],
            "query": each_tweet["query"],
            "user": each_tweet["user"],
            "tweet": each_tweet["tweet"],
            "predicted": predicted if probability > threshold else 2,
            "probability": probability
        }

        li.append(row)    

    # Generate variables for evaluating the model
    final_dataset = pd.DataFrame(li)
    Y_test = final_dataset["polarity"]
    predicted = final_dataset["predicted"]

    model_folder = os.path.join(os.getcwd(), 'saved_models')
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Save evaluations
    sys.stdout = open(os.path.join(model_folder, "confusion_matrix.txt"), "w")
    print("Confusion Matrix:\n", confusion_matrix(Y_test,predicted))
    sys.stdout.close()

    sys.stdout = open(os.path.join(model_folder, "classification_report.txt"), "w")
    print("Classification Report:\n", classification_report(Y_test,predicted))
    sys.stdout.close()

    sys.stdout = open(os.path.join(model_folder, "precision.txt"), "w")
    print("Precision:\n", accuracy_score(Y_test, predicted))
    sys.stdout.close()

    # Save Model
    model_full_path = os.path.join(model_folder, "twitter_sentiment.pk")
    pickle.dump(twitter_sentiment_classifier, open(model_full_path, "wb"))

    # Delete temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)    

if __name__ == "__main__":
    main()