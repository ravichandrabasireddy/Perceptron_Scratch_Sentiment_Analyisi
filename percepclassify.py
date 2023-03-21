#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import string
import re
import json
import sys

def clean_review(review):    
    review = re.sub(r"http\S+", "", review)
    review = re.sub('<.*?>+', '', review)
    review = re.sub('[^A-Za-z]+', ' ', review)
    review = decontracted(review)
    review = remove_punctuation(review)
    review = re.sub("\S*\d\S*", "", review).strip()
    review = review.lower()
    return review

def remove_punctuation(review):
    return ''.join([words for words in review if words not in string.punctuation ])

def decontracted(review):
    review = re.sub(r"won't", "will not", review)
    review = re.sub(r"can\'t", "can not", review)
    review = re.sub(r"n\'t", " not", review)
    review = re.sub(r"\'re", " are", review)
    review = re.sub(r"\'s", " is", review)
    review = re.sub(r"\'d", " would", review)
    review = re.sub(r"\'ll", " will", review)
    review = re.sub(r"\'t", " not", review)
    review = re.sub(r"\'ve", " have", review)
    review = re.sub(r"\'m", " am", review)
    return review

def get_lines_test_data(file_path):
    test_data_lines=[]
    with open(file_path) as perceptron_training_file:
        for lines in perceptron_training_file.readlines():
            test_data_lines.append(lines.rstrip())
    return test_data_lines

stop_words=['me', 'has', 'weren', 'those', 'ours', 'over', 'wasn', 't', 'my', 'theirs', 'having',
           'themselves', 'when', 'he', 'about', 'that', 'as', 'needn', "shan't", 'hers', 'few', 'out', 
           'under', 'now', 'doing', 'ain', "haven't", 'wouldn', 'their', "needn't", 'and', 're', 'had',
           "don't", 'i', 'each', 'very', 'isn', 'its', 'have', 'at', "that'll", 'a', 'by', 'we', 'his',
           'will', "won't", 'between', 'how', 'against', 'but', 'won', 'ma', 'yourselves', 'couldn', 'or',
           'whom', "wasn't", 'll', 'ourselves', 'mustn', 'can', 'after', 'doesn', 'you', 'myself', 'once', 
           'am', "didn't", 'so', 'y', "hadn't", 've', "aren't", 'were', 'because', 'on', 'the', 'they', 'your', 
           'd', 'was', 'our', 'other', 'it', 'until', "it's", 'some', 'aren', 'below', 'here', 'yourself', 
           "hasn't", 'off', 's', 'she', 'this', 'both', 'don', "isn't", 'with', 'too', 'are', 'then', 'o', 
           'didn', 'herself', 'all', 'any', "you'd", 'up', "should've", 'where', 'm', 'nor', 'further', 
           "weren't", 'her', 'into', 'down', 'to', 'shouldn', "you're", 'of', 'yours', 'while', 'who', 
           'again', 'through', 'him', 'most', "doesn't", 'own', 'from', 'for', "mustn't", 'is', 'being',
           'should', 'which', 'them', 'does', 'itself', 'such', 'just', 'no', 'did', "shouldn't", 'same', 
           'than', 'shan', 'before', 'an', "you'll", 'in', 'not', 'do', 'these', 'been', 'himself', 'be', 
           'there', "couldn't", 'above', 'hasn', 'hadn', "mightn't", "wouldn't", 'why', 'haven', 'if', 
           'only', "she's", 'mightn', 'during', 'what', 'more', "you've"]

perceptron_test_data_file_path=sys.argv[2]
lines_in_training_data=get_lines_test_data(perceptron_test_data_file_path)
perceptron_model_file_name=sys.argv[1]
perceptron_model_file=open(perceptron_model_file_name)
perceptron_model=json.load(perceptron_model_file)
word_weight_true_fake_vanila=perceptron_model['word_weight_true_fake_vanila']
bias_true_fake_vanila=perceptron_model['bias_true_fake_vanila']
word_weight_positive_negative_vanila=perceptron_model['word_weight_positive_negative_vanila']
bias_positive_negative_vanila=perceptron_model['bias_positive_negative_vanila']

def total_sum_sign_negative(freqency,word_weight,bias):
    total_sum=0
    for word, freq in freqency.items():
        if word_weight.get(word)!=None:
            total_sum+=(freqency[word] * word_weight[word])
    total_sum+= bias
    if total_sum<=0:
        return True
    return False

freq={}
outputfile=open('percepoutput.txt','w')
for line in lines_in_training_data:
    words=line.split(' ',1)
    review=words[1]
    review=clean_review(review)
    true_words=review.split(' ')
    for word in true_words:
        if word not in stop_words and not word.isspace():
            if freq.get(word)!=None:
                freq[word]+=1
            else:
                freq[word]=1
    if(total_sum_sign_negative(freq,word_weight_true_fake_vanila,bias_true_fake_vanila)):
        outputfile.write(words[0]+" "+"Fake"+" ")
    else:
        outputfile.write(words[0]+" "+"True"+" ")
    if(total_sum_sign_negative(freq,word_weight_positive_negative_vanila,bias_positive_negative_vanila)):
        outputfile.write("Neg"+"\n")
    else:
        outputfile.write("Pos"+"\n")
    freq.clear()


# In[ ]:




