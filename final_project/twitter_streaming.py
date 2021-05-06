import yaml
import os
import requests
import requests_oauthlib
import socket
import json
import sys
from bs4 import BeautifulSoup

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
parameters = yaml.load(open(parameters))
twitter_access_token = parameters["twitter"]["access_token"]
twitter_access_token_secret = parameters["twitter"]["access_token_secret"]
twitter_consumer_key = parameters["twitter"]["api_key"]
twitter_consumer_secret_key = parameters["twitter"]["api_secret_key"]
my_auth = requests_oauthlib.OAuth1(twitter_consumer_key, twitter_consumer_secret_key, twitter_access_token, twitter_access_token_secret)
query = parameters["twitter"]["query"]

def get_tweets():
    url = "https://stream.twitter.com/1.1/statuses/filter.json"
    query_url = ""
    for each_item in query:
        if "track" in each_item:
            for each_term in each_item.get("track"):
                query_url = query_url + "&track=" + str(each_term)
        else:
            query_url = query_url + "&" + str(list(each_item.items())[0][0]) + "=" + str(list(each_item.items())[0][1])

    query_url = url + "?" + query_url[1:]
    response = requests.get(query_url, auth=my_auth, stream=True)
    print(query_url, response)
    return response

def send_tweets_to_spark(http_response, tcp_connection):
    try:
        for each_line in http_response.iter_lines():
            try:
                each_tweet = json.loads(each_line)
                
                row = (
                    (str(each_tweet["id"]) if "id" in each_tweet else "") + "\t" +
                    (str(each_tweet["created_at"]) if "created_at" in each_tweet else "")  + "\t" +
                    str(query) + "\t" +
                    (str(each_tweet["user"]["screen_name"]) if "screen_name" in each_tweet["user"] else "")  + "\t" +
                    BeautifulSoup(each_tweet["text"], "html.parser").get_text().replace("\n", " ") + "\n"
                )
                # print(row)
                tcp_connection.send(row.encode()) 
            except Exception as e:
                if e.errno == 32:
                    print("Error: %s" % e)
                    break
                else:
                    print("Warning (skipping tweet): %s" % e)
    except:
       	e = sys.exc_info()[0]
        print("Error: %s" % e)

tcp_connection = None
tcp_host = parameters["spark"]["host"]
tcp_port = parameters["spark"]["port"]
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

tcp_socket.bind((tcp_host, tcp_port))
tcp_socket.listen(1)

print("Waiting for TCP connection...")
tcp_connection, tcp_address = tcp_socket.accept()

print("Connected! ... Streaming tweets in.")
http_response = get_tweets()
send_tweets_to_spark(http_response, tcp_connection)