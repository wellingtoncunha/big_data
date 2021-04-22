import pandas as pd
import os
import urllib.request 
import shutil
import zipfile
import argparse
import re
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from string import punctuation 
from bs4 import BeautifulSoup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import dill as pickle

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

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

    def buildVocabulary(self, preprocessed_training_dataset):
    # Build vocabulary using all words present on training dataset
        all_words = []
        
        for (words, polarity) in preprocessed_training_dataset:
            all_words.extend(words)

        word_list = nltk.FreqDist(all_words)
        self.word_features = list(word_list.keys())

    def extract_features(self, tweet):
    # Extract features (tag words using on twitter into a instance of dictionary)
        tweet_words=set(tweet)
        self.features={}
        for word in self.word_features:
            self.features['contains(%s)' % word]=(word in tweet_words)
        return self.features 

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

    # Load test and training dataset for exploration
    training_data, testing_data = load_training_dataset(sample_size = sample_size, test_size_frac=test_size_frac)

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