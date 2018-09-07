import nltk
import csv    
import re
import pandas as pd
from pandas import *
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
stemmer = LancasterStemmer()
# 3 classes of training data

df = read_csv("./Training.csv")
df3 = df['Contents']
df4 = df['Tag']
training_data = []

for i in range(0,len(df3)):
	dict1 = {"class" : df4[i],"sentence":df3[i]}
	training_data.append(dict1)
words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern['sentence'])
    # add to our words list
    words.extend(w)
    # add to documents in our corpus
    documents.append((w, pattern['class']))
    # add to our classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)
# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)

# sample training/output
i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w])
print (training[i])
print (output[i])
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
X = np.array(training)
y = np.array(output)

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

# classify("sudo make me a sandwich")
# classify("how are you today?")
# classify("talk to you tomorrow")
# classify("who are you?")
# classify("make me some lunch")
# classify("how was your lunch today?")
# print()
print "----------------------\n"
df = read_csv("./Test_new.csv")
dff = df['Contents'].astype(str)
# data = ["make highly intelligent and accurate decisions for growth Healthcare providers need a partner to turn data into intelligence helping them better understand their market dynamics to create ready to use answers through analytics for constructive effective and efficient growth strategies says Jeff McDonald co founder and president of Expression Health and former SVP of product and platform innovation for Evariant By leveraging our data with the combined data sets from Aegis and Clariture Health and applying our data science techniques we can provide predictive and prescriptive analytics to help all of the Trilliant Health clients build a deeper and richer level of market understanding Jeff and the Expression Health team have developed a powerful analytics platform that ingests and normalizes more than claim lines per month which provides Expression Health s clients with the ability to spot trends and understand patterns in market and provider can deliver market focused and patient level intelligence to assist hospitals and other health care providers in finding retaining and assuming risk for patient populations For more information visit trillianthealth com About Expression Health Analytics Expression Health Analytics helps health care organizations succeed in rapidly changing markets by providing purpose built subscription based analytic insights needed for business growth strategies Using a big data platform over of data data science and interactive business visualizations our analytics answers clients most complex business questions ranging from mergers and acquisitions to new outpatient site development to identifying areas of unmet need without the need for data analysts or sophisticated","first in the marine sector was signed today at the Google Cloud Summit in Sweden It allows Rolls Royce to use Google s Cloud Machine Learning Engine to further train the company s artificial intelligence AI based object classification system for detecting identifying and tracking the objects a vessel can encounter at sea Karno Tenovuo Rolls Royce SVP Ship Intelligence said While intelligent awareness systems will help the latest technology advancements with its deep knowledge of the maritime industry ultimately bringing significant improvements to the sector The Google Cloud Machine Learning Engine uses the same neural net based machine intelligence software which powers many of Google s products including image and voice search Machine Learning is a set of algorithms tools and techniques that mimic human learning to solve specific problems companies will also test whether speech recognition and synthesis are viable solutions for human machine interfaces in marine applications They will also work on optimizing the performance of local neural network computing on board ships using open source machine intelligence software libraries such as Google s TensorFlow Intelligent awareness systems will make vessels safer easier and more efficient to operate by providing crew with an enhanced understanding of their vessel s surroundings"]
for i in dff:
    out = classify(i,show_details = True)
# print out

# classify("make highly intelligent and accurate decisions for growth Healthcare providers need a partner to turn data into intelligence helping them better understand their market dynamics to create ready to use answers through analytics for constructive effective and efficient growth strategies says Jeff McDonald co founder and president of Expression Health and former SVP of product and platform innovation for Evariant By leveraging our data with the combined data sets from Aegis and Clariture Health and applying our data science techniques we can provide predictive and prescriptive analytics to help all of the Trilliant Health clients build a deeper and richer level of market understanding Jeff and the Expression Health team have developed a powerful analytics platform that ingests and normalizes more than claim lines per month which provides Expression Health s clients with the ability to spot trends and understand patterns in market and provider can deliver market focused and patient level intelligence to assist hospitals and other health care providers in finding retaining and assuming risk for patient populations For more information visit trillianthealth com About Expression Health Analytics Expression Health Analytics helps health care organizations succeed in rapidly changing markets by providing purpose built subscription based analytic insights needed for business growth strategies Using a big data platform over of data data science and interactive business visualizations our analytics answers clients most complex business questions ranging from mergers and acquisitions to new outpatient site development to identifying areas of unmet need without the need for data analysts or sophisticated", show_details=True) 