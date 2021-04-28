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
training_data = spark.read.load(
    "tmp/training.1600000.processed.noemoticon.csv",
    format="csv")
training_data = training_data.withColumnRenamed("_c0", "label") \
    .withColumnRenamed("_c1", "tweet_id") \
    .withColumnRenamed("_c2", "date") \
    .withColumnRenamed("_c3", "query") \
    .withColumnRenamed("_c4", "user") \
    .withColumnRenamed("_c5", "tweet")

sample_size = 10000
training_data = training_data.sample(sample_size / 1600000)

training_data = training_data.select(functions.col("label"), functions.col("tweet"))

udf_cleansing_and_tokenizing = functions.udf(cleansing_and_tokenizing)
training_data = training_data.withColumn("tweet_cleansed", udf_cleansing_and_tokenizing(functions.col("tweet")))
training_data = training_data.withColumn("tweet_cleansed", functions.split("tweet_cleansed", " ")) 

#training_data.show(5)

vocabulary = training_data.withColumn("word", functions.explode("tweet_cleansed")).select(functions.col("word"))
#print("Count:", vocabulary.count())
vocabulary = vocabulary.distinct()
#print("Distinct count:", vocabulary.count())
vocabulary = vocabulary.withColumn("dummy_col", functions.lit(1))
vocabulary_list = vocabulary.groupBy("dummy_col").agg(functions.collect_list("word"))
vocabulary_list = vocabulary_list.withColumnRenamed("collect_list(word)", "words")

training_data = training_data.join(vocabulary_list.select("words"))
#df.withColumnRenamed('id', 'id1').crossJoin(df.withColumnRenamed('id', 'id2')).show()
#training_data.show(5)

def extract_features(tweet, vocabulary):
# Extract features (tag words using on twitter into a instance of dictionary)
    tweet_words=set(tweet)
    features={}
    for word in vocabulary:
        features['contains(%s)' % word]=(word in tweet_words)
    return features 

udf_extract_features = functions.udf(extract_features)
training_data = training_data.withColumn(
    "features", 
    udf_extract_features(functions.col("tweet"), functions.col("words"))
)

#training_data.show(5)
training, test = training_data.select("features", "label").randomSplit([0.5, 0.5])

#training.show(5)

training_features = training.rdd.map(tuple).collect()
training_features

# import nltk
# Classifier = nltk.NaiveBayesClassifier.train(training_features)