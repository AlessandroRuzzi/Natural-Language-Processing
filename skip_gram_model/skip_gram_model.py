V_size = 12000 #size of the vocabulary
N = 300 #embedding size
window_size = 5
K= 20 #number of negative samples per positive pairs (wt,wi)
Step_size = 500000

import nltk
import numpy as np
nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import brown
from nltk.corpus import stopwords
from numpy import save
from matplotlib import pyplot as plt

np.random.seed(42)

sentences = brown.sents()
print(len(sentences))
print(sentences[:3])

import string
stop_words = set(stopwords.words('english'))
def preprocessing(sentences):
    training_data = []
    for i in range(len(sentences)):
        #sentences[i] = sentences[i].strip()
        sentence = sentences[i]
        x = [word.strip(string.punctuation) for word in sentence]
        x = [word.lower() for word in x]
        x = [word for word in x if word!='' and word.isalpha()]
        x = [word for word in x if word not in stop_words]
        if x:
          training_data.append(x)
    return training_data
sentences = preprocessing(sentences)
print(sentences[:3])

from collections import defaultdict
count = defaultdict(int)
for sentence in sentences:
    for word in sentence:
      count[word] += 1

import operator
sorted_counts = sorted(count.items(), key=operator.itemgetter(1), reverse=True)


vocabulary = sorted_counts[:V_size]

# Assign ids and create lookup tables
word_to_index = {'UNK': 0}
index_to_word = ['UNK']
for idx, tup in enumerate(vocabulary,1):
  word_to_index[tup[0]] = idx
  index_to_word.append(tup[0])

assert len(index_to_word) == len(word_to_index)
print(len(index_to_word))

sentences_index = []
for sent in sentences:
  ids=[]
  for word in sent:
    if word in word_to_index:
      ids.append(word_to_index[word])
    else:
      ids.append(0)
  sentences_index.append(ids)
print(len(sentences_index))
print(sentences_index[:3])

dataset_tmp = []
for sentence in sentences_index:
    for i in range(len(sentence)):
        index = 1
        while index <= window_size and i-index>=0:  #sliding window to the left of the word
            dataset_tmp.append(tuple((sentence[i],sentence[i-index])))
            index +=1
        index = 1
        while index <= window_size and i + index < len(sentence): #sliding window to the right of the word
            dataset_tmp.append(tuple((sentence[i], sentence[i +index])))
            index += 1
print(len(dataset_tmp))
print(dataset_tmp[:15])

dataset = []
count = 0
probs = list(np.array(list(zip(*vocabulary))[1])/sum(list(zip(*vocabulary))[1]))  #calculate probabilities of words to be drawn
words_to_sample = list(list(zip(*vocabulary))[0])  #create the list of the words to draw

for i in range(len(words_to_sample)):
    words_to_sample[i] = word_to_index[words_to_sample[i]]   #convert words into indeces
words_to_sample = tuple(words_to_sample)

negative_samples = np.random.choice(size = K*len(dataset_tmp),a= words_to_sample, p=probs)  #sample the words only one time to optimize the code
print(len(words_to_sample))
for i in range(len(dataset_tmp)):
    if(count % Step_size == 0):
        print(f"sample {count}/{len(dataset_tmp)}")
    sample_list = []
    sample_list.append(dataset_tmp[i][0])
    sample_list.append(dataset_tmp[i][1])
    index = K*i
    if dataset_tmp[i][1] in negative_samples[index:index+20]: #check if the positive word is present in the list of negative words
        for j in range(K):
            if(negative_samples[index+j] == dataset_tmp[i][1]):
                negative_sample = negative_samples[index+j]
                while(negative_sample == dataset_tmp[i][1]):  #sample another word to replace the positive word
                    negative_sample = np.random.choice(a=words_to_sample, p=probs)
                negative_samples[index+j] = negative_sample


    to_add_list = list(negative_samples[index:index+20])
    for element in to_add_list:  #create the sample with positive word, and negative words
      sample_list.append(element)
    count +=1
    dataset.append(sample_list)

print(dataset[:3])
print(count)


#training params
lr = 0.03
epochs = 10


W = np.random.uniform(-0.8, 0.8,size=(V_size+1,N))
W_prime = np.random.uniform(-0.8, 0.8,size=(V_size+1,N))

import math


def sigmoid(x):
  return 1 / (1 + math.e ** -x)


def wrd_gradient(W,W_prime,sample):  # calculate the word gradient
    first_term = -(1-sigmoid(np.dot(W[sample[0]],W_prime[sample[1]])))*W_prime[sample[1]]
    second_term = 0
    for i in range(K):
        second_term += (sigmoid(np.dot(W[sample[0]],W_prime[sample[2+i]])))*W_prime[sample[2+i]]

    return first_term + second_term


def ctx_gradient(W,W_prime,sample,negative = False,index = 0):  # calculate the context gradient for both negative and postive words
    if negative:
        gradient = (sigmoid(np.dot(W[sample[0]], W_prime[sample[2+index]]))) * W[sample[0]]
    else:
        gradient = -(1-sigmoid(np.dot(W[sample[0]],W_prime[sample[1]])))*W[sample[0]]

    return gradient


def loss_one_sample(W,W_prime,sample):  # calculate the loss for a single sample
    first_term = -np.log(sigmoid(np.dot(W[sample[0]], W_prime[sample[1]])))
    second_term = 0
    for i in range(K):
        second_term += - np.log(1 - sigmoid(np.dot(W[sample[0]], W_prime[sample[2 + i]])))

    return first_term + second_term

##fit
total_loss = 0
counter = 0

steps_loss = []

for epoch in range(epochs):
    print(f"epoch: {epoch + 1}/{epochs}")
    np.random.shuffle(dataset)  # shuffle the datatset at each iteration
    dataset = list(dataset)
    counter = 0
    for sample in dataset:
        if (counter % Step_size == 0):
            total_loss = 0
            print(f"sample: {counter}/{len(dataset)}")

        steps_loss.append(loss_one_sample(W, W_prime, sample))  # calculate the sample loss
        gradient_wrd = wrd_gradient(W,W_prime,sample)

        gradient_ctx = ctx_gradient(W, W_prime, sample)
        gradient_ctx_neg = []

        # Steps of stochastic gradient descent
        for i in range(K):
            gradient_ctx_neg.append(ctx_gradient(W, W_prime, sample,negative=True,index = i))
        W[sample[0]] = W[sample[0]] - lr*gradient_wrd
        W_prime[sample[1]] = W_prime[sample[1]] - lr * gradient_ctx
        for i in range(K):
            W_prime[sample[2+i]] = W_prime[sample[2+i]] - lr * gradient_ctx_neg[i]
        counter +=1



import pandas as pd

# Calculate the running average loss
average_windows_size = 100000
numbers_series = pd.Series(steps_loss)
windows = numbers_series.rolling(average_windows_size)
moving_averages = windows.mean()
moving_averages_list = moving_averages.tolist()
without_nans = moving_averages_list[window_size - 1:]

# Plot The running average loss
plt.plot(without_nans)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.yticks(np.arange(0,12,0.5))
plt.ylim(top=12,bottom=2.5)
plt.grid(True)
plt.title('Running average loss')
plt.savefig("loss_skip_gram.png")
plt.show()

print("train end")

# Save both the matrices
save("W.npy",W)
save("W_prime.npy",W_prime)


