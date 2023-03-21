#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import string
import re
import json

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

def get_lines_training_data(file_path):
    training_data_lines=[]
    with open(file_path) as hmm_training_file:
        for lines in hmm_training_file.readlines():
            training_data_lines.append(lines.rstrip())
    return training_data_lines

def perceptron_train(freqency,word_weight,bias,sign):
    total_sum=0
    for word, freq in freqency.items():
        total_sum+=(freq * word_weight[word])
    total_sum+= bias
    if sign*total_sum<=0:
        for word, freq in freqency.items():
            word_weight[word]+=(sign*freqency[word])
        bias+=sign
    return word_weight,bias

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



perceptron_training_data_file_path="perceptron-training-data/train-labeled.txt"
lines_in_training_data=get_lines_training_data(perceptron_training_data_file_path)

review_true_fake=[]
review_positive_negative=[]

word_weight_true_fake_vanila,word_weight_positive_negative_vanila={},{}
word_weight_true_fake_averaged,word_weight_positive_negative_averaged={},{}
update_vanila,update_averaged={},{}
bias_true_fake_vanila,bias_positive_negative_vanila=0,0
bias_true_fake_averaged,bias_positive_negative_averaged=0.0,0.0
sum_true_fake_averaged,sum_positive_negative_averaged=0.0,0.0
frequency_count=[]
freq={}

for line in lines_in_training_data:
    words=line.split(' ',3)
    if words[1]=='True':
        review_true_fake.append(1)
    else:
        review_true_fake.append(-1)
    if words[2]=="Pos":
        review_positive_negative.append(1)
    else:
        review_positive_negative.append(-1)
    review=words[3]
    review=clean_review(review)
    true_words=review.split(' ')
    for word in true_words:
        if word not in stop_words and not word.isspace():
            if freq.get(word)!=None:
                freq[word]+=1
            else:
                word_weight_true_fake_vanila[word],word_weight_positive_negative_vanila[word],word_weight_true_fake_averaged[word],word_weight_positive_negative_averaged[word]=0,0,0.0,0.0
                update_vanila[word],update_averaged[word]=0.0,0.0
                freq[word]=1
    frequency_count.append(freq.copy())
    freq.clear()    


epochs=30
for iteration in range(epochs):
    for index in range(len(frequency_count)):
        word_weight_true_fake_vanila,bias_true_fake_vanila=perceptron_train(frequency_count[index],word_weight_true_fake_vanila,bias_true_fake_vanila,review_true_fake[index])
        word_weight_positive_negative_vanila,bias_positive_negative_vanila=perceptron_train(frequency_count[index],word_weight_positive_negative_vanila,bias_positive_negative_vanila,review_positive_negative[index])
        

with open('vanila_perceptron.txt','w') as file:
    file.write(json.dumps({"word_weight_true_fake_vanila":word_weight_true_fake_vanila,"bias_true_fake_vanila":bias_true_fake_vanila,"word_weight_positive_negative_vanila":word_weight_positive_negative_vanila,"bias_positive_negative_vanila":bias_positive_negative_vanila}))


# In[ ]:




