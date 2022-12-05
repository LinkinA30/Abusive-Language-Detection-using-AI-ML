#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:31:51 2020

@author: rohan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tweepy import API 
from tweepy import Cursor
from tweepy import OAuthHandler
import datetime
import csv
import nltk

import matplotlib.pyplot as plt
import tweepy
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#from sklearn.externals import joblib
import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from openpyxl import load_workbook

nltk.download('stopwords')
ps=PorterStemmer()

ACCESS_TOKEN ="1053329844635873283-PhypsHaZMv0e4ufDq3Yd9OXBpMW2Lx"
ACCESS_TOKEN_SECRET = "2yKkax4j8NRmvY00kQuqdxJOE2bdeHWmWTgnlvEJIuKHf"
CONSUMER_KEY = "f2zFtlM6qyJIEgGAZJ8yvBTwS"
CONSUMER_SECRET = "RcUQ1FjOKYkBJFILzTc1I1Zbhiq0vafjmERnKTczYMbGcqoUzb"

tfidf_transformer = TfidfTransformer()
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=100,stop_words='english',analyzer='word',lowercase=True)#,token_pattern='[^a-zA-Z]')


def train():
    df = pd.read_csv("dataset//stemmed.csv")
    df.astype('U')
    
    X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['class'],test_size=0.2)
    X_train_counts = count_vect.fit_transform(X_train)
    
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_tfidf,y_train)
    
    import sklearn.model_selection as ms
    seed=7
    kfold = ms.KFold(n_splits=10)
    results = ms.cross_val_score(clf,X_train_tfidf , y_train, cv=kfold)
    print('K-fold(10) cross validation results:\n',results)
    
    #Predictions
    X_new_counts = count_vect.transform(X_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)#use the tfidif or countvectorizer to do the predictions
    
    from sklearn.metrics import confusion_matrix
    import sklearn.metrics as mt
    print("Accuracy: %.3f%% " % (mt.accuracy_score(y_test,predicted)*100.0))
    print('Confusion matrix:',confusion_matrix(y_test,predicted))
    np.mean(predicted == y_test)       
    
    joblib.dump(clf,'nlp_model.pkl')
    print('Model Saved')
    
    
    
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
        return auth
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets
    


class TweetAnalyzer():
   
   # Functionality for analyzing and categorizing content from tweets.

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
            
    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])
        return df

def test():
    twitter_client = TwitterClient()
    nlp = joblib.load('nlp_model.pkl')

    api = twitter_client.get_twitter_client_api()
    auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth,wait_on_rate_limit=True)

    #tweets = api.user_timeline(screen_name="realDonaldTrump", count=20)
    tweets_in=["Virat is an asshole", "RCB will never win"]
    tweets=[]
    #df = tweet_analyzer.tweets_to_data_frame(tweets)
    #has=input('Enter the hashtag to curate abusive contents:')
    #for tweet in tweepy.Cursor(api.search,q=has,count=1,lang="en").items():
    for tw in tweets_in:

        #tw=tweet.text
        review=re.sub('[^a-zA-Z]',' ',tw)      
        review=review.lower()          
        review=review.split()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
        review=' '.join(review)

        tweets.append(review)
        print(tw)
    #csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    
    
    
    df1 = pd.DataFrame(data=[t for t in tweets], columns=['tweets'])
    test1 = df1['tweets']
    
    results = []
    count = 0
    resultFeed=[]
    for tw in test1:
        
        tw1=[tw]
        test_cv = count_vect.transform(tw1)
        
        test_tfidf = tfidf_transformer.transform(test_cv)
        test_predicted = nlp.predict(test_tfidf)
        #print(tw,test_predicted)
        results.append(test_predicted)
        if test_predicted == 0:
            resultFeed.append('Neutral')
        if test_predicted == 1:
            resultFeed.append('Abusive')
        count = count + 1
    
    abusiveCount=0
    neutralCount=0
    dateList = [datetime.date.today()] * len(resultFeed)
    
    for result in results:
        if result == 0:
            neutralCount=neutralCount+1
        else:
            abusiveCount = abusiveCount + 1
            
    sizes = [neutralCount,abusiveCount]
    labels=['Neutral '+str(neutralCount),'Abusive '+str(abusiveCount)]
    colors = ['yellowgreen', 'gold']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    #plt.title('Analysis of #%s'%(has))
    plt.tight_layout()
    plt.show()
    
    filename = has+'.csv'
    csvFile = open(filename, 'a',newline='')
    csvWriter = csv.writer(csvFile)
    for count in range(0,len(test1)):
        csvWriter.writerow([test1[count],resultFeed[count],dateList[count]])
    csvFile.close()