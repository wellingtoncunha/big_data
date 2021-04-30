import os
import shutil
import zipfile

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

def cleansing_and_tokenizing(tweet):
# Cleansing and tokenizing tweet
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

unzip_files()

from pyspark.sql import SparkSession, functions


spark = SparkSession.builder.master("local").appName("Training Twitter Sentiment Analysis").getOrCreate()
test_data = spark.read.load(
    "tmp/testdata.manual.2009.06.14.csv",
    format="csv")
test_data = test_data.withColumnRenamed("_c0", "label") \
    .withColumnRenamed("_c1", "tweet_id") \
    .withColumnRenamed("_c2", "date") \
    .withColumnRenamed("_c3", "query") \
    .withColumnRenamed("_c4", "user") \
    .withColumnRenamed("_c5", "tweet")
test_data = test_data.withColumn("label", functions.col("label").cast("integer"))

udf_cleansing_and_tokenizing = functions.udf(cleansing_and_tokenizing)
test_data = test_data.withColumn("tweet_cleansed", udf_cleansing_and_tokenizing(functions.col("tweet")))

from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="tweet_cleansed", outputCol="words")
test_data = tokenizer.transform(test_data)

from pyspark.ml.feature import HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="term_freq")
test_data = hashingTF.transform(test_data)
test_data.show(5)

from pyspark.ml.feature import IDF 
idf = IDF(inputCol="term_freq", outputCol="tfidf")
idfModel = idf.fit(test_data)
test_data = idfModel.transform(test_data)
test_data.show(5)

from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol="label", outputCol="labelIndex")
model = stringIndexer.fit(test_data)
test_data = model.transform(test_data)
test_data.show(5)

predicted = test_data.select("tfidf", "labelIndex")
predicted.show(5)

model_folder = os.path.join(os.getcwd(), 'saved_models')
model_full_path = os.path.join(model_folder, "twitter_sentiment_spark")
if not os.path.exists(model_folder):
    print("model does not exists")

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
loadModel = NaiveBayesModel.load(model_full_path)
predicted = loadModel.transform(predicted)
predicted.show()

total = predicted.count()
correct = predicted.where(predicted['labelIndex'] == predicted['NB_pred']).count()
accuracy = correct/total

print(
    "\nTotal:", total, 
    "\nCorrect:", correct, 
    "\nAccuracy:", accuracy)