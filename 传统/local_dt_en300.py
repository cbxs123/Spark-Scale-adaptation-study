#-*- coding:UTF-8 -*-
import os
import io
import sys
import imp
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import tree  
from sklearn.svm import SVC
from sklearn import neighbors  
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
import time
from sklearn.cross_validation import train_test_split
import csv


# 配置utf-8输出环境  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')


def pre_data():
    fo_label = open('label.txt', "w",encoding='UTF-8')

    with open("/home/ubuntu/ys-180326/Dataset300.csv","r",encoding='UTF-8') as csvFile:
        reader = csv.DictReader(csvFile)
       
        label_str =   [row['Sentiment'] for row in reader]
        print(len(label_str))
        for i in label_str:
            fo_label.write(i+"\n")
    fo_label.close()
    
    fo_content = open('content.txt', "w",encoding='UTF-8')     
    with open("/home/ubuntu/ys-180326/Dataset300.csv","r",encoding='UTF-8') as csvFile:
        reader = csv.DictReader(csvFile)
       
        content =   [row['SentimentText'] for row in reader]
        print(len(content))
        for i in content:
            fo_content.write(i.strip()+"\n")
    fo_content.close()

def readDataSet():
    neg_content = []
    neg_label = []
    pos_content = []
    pos_label = []
    
    label = []
    content = []
    with open("/home/ubuntu/ys-180326/Dataset300.csv","r",encoding='UTF-8') as csvFile:
        reader = csv.DictReader(csvFile)
       
        label_str =   [row['Sentiment'] for row in reader]
        for i in label_str:
            label.append(int(i))
        
    with open("/home/ubuntu/ys-180326/Dataset300.csv","r",encoding='UTF-8') as csvFile:
        reader = csv.DictReader(csvFile)
       
        content =   [row['SentimentText'] for row in reader]
        
    #for i in range(len(label)):
    #    print(label[i]+'!!!'+content[i])

    train_comment, test_comment, train_label,test_label =  train_test_split(content, label, test_size=0.2)    
        
    
    
    
   
    return train_comment , train_label , test_comment ,test_label

    
def local(train_comment, train_label, test_comment, test_label, clf):
    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()
    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_comment))  
    
    time_start = time.time()
    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', clf)])
    text_clf = text_clf.fit(train_comment, train_label)
    time_end = time.time()
    
    predicted = text_clf.predict(test_comment)
    
    print(time_end - time_start)
    print(np.mean(predicted == test_label))
    print(metrics.accuracy_score(test_label, predicted))
    print(metrics.recall_score(test_label, predicted))
    print(metrics.f1_score(test_label, predicted, average='weighted'))

    #print(set(predicted))
    
    
if __name__ == '__main__':
   
    train_comment , train_label , test_comment ,test_label = readDataSet()
    clf_dt =  tree.DecisionTreeClassifier(criterion='gini')
    local(train_comment , train_label , test_comment ,test_label, clf_dt)
   
  
    
   

    
