

cd ~/Downloads
git clone https://github.com/wellingtoncunha/big_data.git
sudo rm -rf /app/twitter_sentiment_analysis
sudo mkdir -p /app/twitter_sentiment_analysis
sudo chmod a+rwx /app/twitter_sentiment_analysis

cd big_data/final_project
sudo cp -r ~/Downloads/big_data/final_project/training_dataset/ /app/twitter_sentiment_analysis/
sudo cp ~/Downloads/big_data/final_project/training_model_spark.py /app/twitter_sentiment_analysis/
touch /app/twitter_sentiment_analysis/parameters.yaml
nano /app/twitter_sentiment_analysis/parameters.yaml


sudo apt install python3 -y
sudo apt install python3-pip -y
pip3 install pyspark

cd /app/twitter_sentiment_analysis/
python3 /app/twitter_sentiment_analysis/training_model_spark.py
spark-submit --master yarn --deploy-mode cluster  py_files.py /app/twitter_sentiment_analysis/training_model_spark.py

import pip3 install requests_oauthlib
sudo cp ~/Downloads/big_data/final_project/twitter_sentiment_analysis_spark.py /app/twitter_sentiment_analysis/
