import os
import parser
import shutil
import sys
import dill as pickle
import argparse
import zipfile
import pandas as pd
from bs4 import BeautifulSoup
import re
from string import punctuation 
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import twitter
import yaml
import json
import unidecode
import time

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

def load_test_dataset():
# Unzip and load test dataset
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)
        
    if not os.path.exists(temporary_folder):
        os.makedirs(temporary_folder)
        
    local_file_name = os.path.join(base_folder, "training_dataset", "trainingandtestdata.zip")
    with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
        zip_ref.extractall(temporary_folder)
    
    test_dataset_path = os.path.join(
        temporary_folder, 
        "testdata.manual.2009.06.14.csv")

    test_dataset = pd.read_csv(
        test_dataset_path, 
        encoding="latin-1", 
        warn_bad_lines=True,
        error_bad_lines=False,
        header=None, 
        names=["polarity", "tweet_id", "date", "query", "user", "tweet"])

    # Delete temporary folder
    if os.path.exists(temporary_folder):
        shutil.rmtree(temporary_folder)    

    return test_dataset.to_dict("records")

def classify_tweets(inbound_dataset, probability_threshold):
    model_folder = os.path.join(os.getcwd(), 'saved_models')
    model_full_path = os.path.join(model_folder, "twitter_sentiment.pk")
    twitter_sentiment_classifier = pickle.load(open(model_full_path, "rb"))
    li = []
    threshold = probability_threshold
    for each_tweet in inbound_dataset:    
        words = twitter_sentiment_classifier.processTweet(each_tweet["tweet"])
        features = twitter_sentiment_classifier.extract_features(words)
        predicted = twitter_sentiment_classifier.Classifier.classify(features)
        probability = twitter_sentiment_classifier.Classifier.prob_classify(features).prob(predicted)
        if "polarity" in each_tweet:
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
        else:
           row = {
                "tweet_id": each_tweet["tweet_id"],
                "date": each_tweet["date"],
                "user": each_tweet["user"],
                "tweet": each_tweet["tweet"],
                "predicted": predicted if probability > threshold else 2,
                "probability": probability
            }
        li.append(row)  

    return pd.DataFrame(li)

def get_twitter(search_keyword, fetch_size):
    parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
    parameters = yaml.load(open(parameters))
    twitter_api = twitter.Api(
        consumer_key = parameters["api_key"],
        consumer_secret = parameters["api_secret_key"],
        access_token_key = parameters["access_token"],
        access_token_secret= parameters["access_token_secret"],
        tweet_mode="extended"
    )
    print(twitter_api.VerifyCredentials())

    tweets_fetched = twitter_api.GetSearch(search_keyword, lang = "en", count = fetch_size)
    
    print("Fetched " + str(len(tweets_fetched)) + " tweets for the term " + search_keyword)
    li = []
    for each_tweet in tweets_fetched:
        each_tweet_tmp = (str(each_tweet))
        each_tweet_tmp = json.loads(each_tweet_tmp)
        row = {
            "tweet_id": each_tweet_tmp["id"],
            "date": each_tweet_tmp["created_at"],
            "user": each_tweet_tmp["user"]["screen_name"],
            "tweet": unidecode.unidecode(each_tweet_tmp["full_text"])
        }
        li.append(row)
    twitter_dataset = pd.DataFrame(li)

    return twitter_dataset.to_dict("records") 

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
            inbound_dataset = load_test_dataset()
            outbound_dataset = classify_tweets(inbound_dataset, probability_threshold)
            Y_test = outbound_dataset["polarity"]
            predicted = outbound_dataset["predicted"]
            test_folder = os.path.join(os.getcwd(), 'test_model')    
            if not os.path.exists(test_folder):
                os.makedirs(test_folder)

            # Save evaluations
            sys.stdout = open(os.path.join(test_folder, "confusion_matrix.txt"), "w")
            print("Confusion Matrix:\n", confusion_matrix(Y_test,predicted))
            sys.stdout.close()

            sys.stdout = open(os.path.join(test_folder, "classification_report.txt"), "w")
            print("Classification Report:\n", classification_report(Y_test,predicted))
            sys.stdout.close()

            sys.stdout = open(os.path.join(test_folder, "precision.txt"), "w")
            print("Precision:\n", accuracy_score(Y_test, predicted))
            sys.stdout.close()

            # Save Dataset
            file_name = os.path.join(test_folder, "outbound_test.csv")
            outbound_dataset.to_csv(file_name, index=False)
        else:
            if "search_keyword" not in args:
                print("A search keyword must be informed!")
                sys.exit()
            else:
                search_keyword = args.search_keyword
                if search_keyword.strip() == "": 
                    print("A search keyword must be informed!")
                    sys.exit()
            if "fetch_size" not in args:
                print("The size of fetch was not informed! Default (100) will be used")
                fetch_size = 100
            else:
                fetch_size = args.fetch_size

            inbound_dataset = get_twitter(search_keyword, fetch_size)
            outbound_dataset = classify_tweets(inbound_dataset, probability_threshold)

            output_folder = os.path.join(os.getcwd(), 'outbound')    
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            file_name = "outbound_" + time.strftime("%Y%m%d_%H%M%S") + ".csv"
            file_name = os.path.join(output_folder, file_name)
            outbound_dataset.to_csv(file_name, index=False)



if __name__ == "__main__":
    main()