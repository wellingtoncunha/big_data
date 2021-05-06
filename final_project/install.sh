# clones the app from Github
clear
cd ~/Downloads
rm -rf ~/Downloads/big_data
git clone https://github.com/wellingtoncunha/big_data.git

# the following statement should be used with caution, because it will empty the installation folder, including any saved model
# sudo rm -rf /app/twitter_sentiment_analysis

# creates a folder to hold the scripts, configuration file and training dataset
sudo mkdir -p /app/twitter_sentiment_analysis
sudo chmod a+rwx /app/twitter_sentiment_analysis

# Copy required files to the installation folder
cd big_data/final_project
sudo cp -r ~/Downloads/big_data/final_project/training_dataset/ /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/training_model_spark.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/twitter_streaming.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/twitter_sentiment_analysis_spark.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/mongo-spark-connector_2.11-2.4.3.jar /app/twitter_sentiment_analysis/
touch /app/twitter_sentiment_analysis/parameters.yaml
nano /app/twitter_sentiment_analysis/parameters.yaml # Use the parameters_template.yaml to create your own

## Training script
# Install pip packages required by Python script
sudo apt install python3 -y
sudo apt install python3-pip -y
pip3 install pyspark
pip3 install numpy
pip3 install nltk
pip3 install bs4
pip3 install pandas --upgrade

# Train the model (it takes around 2 hours for the whole training dataset with 1.6MM or rows)
cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/training_model_spark.py

## Classification (sentiment analysis) script
# Install pymongo pip package
pip3 install pymongo

# Start the classification on streming mode - it requires that the stream in app to be running
cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/twitter_sentiment_analysis_spark.py

## Stream in script
# Install required pip package
pip3 install requests_oauthlib

# Run the stream in application. It requires the sentiment analysis script to be running, otherwise it goes to a "waiting mode"
cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/twitter_streaming.py

# this is a snippet if we need to free the port after any interruption (caused by error or manual interruption)
sudo fuser -k 9009/tcp
