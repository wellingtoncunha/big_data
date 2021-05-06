
clear
cd ~/Downloads
rm -rf ~/Downloads/big_data
git clone https://github.com/wellingtoncunha/big_data.git
#sudo rm -rf /app/twitter_sentiment_analysis
#sudo mkdir -p /app/twitter_sentiment_analysis
#sudo chmod a+rwx /app/twitter_sentiment_analysis

cd big_data/final_project
sudo cp -r ~/Downloads/big_data/final_project/training_dataset/ /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/training_model_spark.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/twitter_streaming.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/twitter_sentiment_analysis_spark.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/mongo-spark-connector_2.11-2.4.3.jar /app/twitter_sentiment_analysis/
touch /app/twitter_sentiment_analysis/parameters.yaml
nano /app/twitter_sentiment_analysis/parameters.yaml


sudo apt install python3 -y
sudo apt install python3-pip -y
pip3 install pyspark
pip3 install numpy
pip3 install nltk
pip3 install bs4
pip3 install pandas --upgrade

cd /app/twitter_sentiment_analysis/
# sudo nano python3 /app/twitter_sentiment_analysis/training_model_spark.py
python3 /app/twitter_sentiment_analysis/training_model_spark.py
# spark-submit --master yarn --deploy-mode cluster  /app/twitter_sentiment_analysis/training_model_spark.py

pip3 install pymongo

cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/twitter_sentiment_analysis_spark.py

pip3 install requests_oauthlib

cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/twitter_streaming.py

db.sentiment_analysis.find().limit(1).pretty();
db.sentiment_analysys.findOne().sort({x:1}).pretty();


sudo fuser -k 9009/tcp

sudo cp ~/Downloads/big_data/final_project/twitter_streaming.py /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/twitter_sentiment_analysis_spark.py /app/twitter_sentiment_analysis/


%SPARK_HOME%\python;%SPARK_HOME%\python\lib\py4j-<version>-src.zip:%PYTHONPATH%

export PYSPARK_PYTHON=python3

http://54.158.162.208:8088/cluster/app/application_1620224092720_0002
http://54.158.162.208:8088/proxy/application_1620224092720_0028