import os
import parser
import shutil
import sys

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
    for each_tweet in testing_data:    
        words = twitter_sentiment_classifier.processTweet(each_tweet["tweet"])
        features = twitter_sentiment_classifier.extract_features(words)
        predicted = twitter_sentiment_classifier.Classifier.classify(features)
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

    return pd.DataFrame(li)

def main():
    parser = argparse.ArgumentParser(description="Twitter sentiment analysis classification")
    parser.add_argument("--probability_threshold", type=float, default=None, 
        help="an numeric between 0.5 and 1 that will be used as a threshold to classify the tweet. If probability is lower than it, then the twitter is classified as neutral (polarity=2)")
    parser.add_argument("--test", type=bool, default=None, 
        help="run the classification for the test file available and save it to /test folder")
    args = parser.parse_args()
    if "probability_threshold" in args:
        probability_threshold = args.probability_threshold
    else:
        probability_threshold = 0

    
    if "test" in args:
        inbound_dataset = load_test_dataset():
        outbound_dataset = classify_tweets(inbound_dataset, probability_threshold)
        Y_test = outbound_dataset["polarity"]
        predicted = outbound_dataset["predicted"]
        test_folder = os.path.join(os.getcwd(), 'test_model')    
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

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
        outbound_dataset.to_csv(index=False)

    else:
        inbound_dataset = None


if __name__ == "__main__":
    main()