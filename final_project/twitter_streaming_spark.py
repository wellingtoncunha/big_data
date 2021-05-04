import yaml
import os
import requests
import requests_oauthlib
import socket
import json

base_folder = os.getcwd()
temporary_folder = os.path.join(os.getcwd(), 'tmp')

parameters = os.path.abspath(os.path.join(base_folder, "parameters.yaml"))
parameters = yaml.load(open(parameters))
ACCESS_TOKEN = parameters["twitter"]["access_token"]
ACCESS_SECRET = parameters["twitter"]["access_token_secret"]
CONSUMER_KEY = parameters["twitter"]["api_key"]
CONSUMER_SECRET = parameters["twitter"]["api_secret_key"]
my_auth = requests_oauthlib.OAuth1(CONSUMER_KEY, CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_SECRET)
query = parameters["twitter"]["query"]

def get_tweets():
    url = 'https://stream.twitter.com/1.1/statuses/filter.json'	
    query_data = [('language', 'en'), ('locations', '-130,-20,100,50')]
    query_url = url + '?' + '&'.join([str(t[0]) + '=' + str(t[1]) for t in query_data])
    query_url = query_url + "&" + '&'.join(['track=' + str(t) for t in query])
    response = requests.get(query_url, auth=my_auth, stream=True)
    print(query_url, response)
    return response

def send_tweets_to_spark(http_resp, tcp_connection):
    for each_line in http_resp.iter_lines():
        each_tweet = json.loads(each_line)
        row = (
            str(each_tweet["id"]) + "\t" +
            str(each_tweet["created_at"]) + "\t" +
            str(each_tweet["user"]["screen_name"]) + "\t" +
            str(each_tweet["text"]) + "\n"
        )
        tcp_connection.send(row.encode()) 

TCP_IP = parameters["spark"]["host"]
TCP_PORT = parameters["spark"]["port"]
conn = None
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind((TCP_IP, TCP_PORT))
s.listen(1)

print("Waiting for TCP connection...")
conn, addr = s.accept()

print("connected... Starting getting tweets.")
resp = get_tweets()
send_tweets_to_spark(resp, conn)