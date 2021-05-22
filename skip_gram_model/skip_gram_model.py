V_size = 12000 #size of the vocabulary
N = 300 #embedding size
window_size = 5
K= 20 #number of negative samples per positive pairs (wt,wi)

import nltk
import numpy as np
nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import brown
from nltk.corpus import stopwords

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
#print(sorted_counts)

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
        while index <= window_size and i-index>=0:
            dataset_tmp.append(tuple((sentence[i],sentence[i-index])))
            index +=1
        index = 1
        while index <= window_size and i + index < len(sentence):
            dataset_tmp.append(tuple((sentence[i], sentence[i +index])))
            index += 1
print(len(dataset_tmp))
print(dataset_tmp[:15])

dataset = []
count = 0
probs = list(np.array(list(zip(*vocabulary))[1])/sum(list(zip(*vocabulary))[1]))
words_to_sample = list(list(zip(*vocabulary))[0])

for i in range(len(words_to_sample)):
    words_to_sample[i] = word_to_index[words_to_sample[i]]
words_to_sample = tuple(words_to_sample)

negative_samples = np.random.choice(size = K*len(dataset_tmp),a= words_to_sample, p=probs)

for i in range(len(dataset_tmp)):
    if(count % 100000 == 0):
        print(f"sample {count}/{len(dataset_tmp)}")
    sample_list = []
    sample_list.append(dataset_tmp[i][0])
    sample_list.append(dataset_tmp[i][1])
    index = K*i
    if dataset_tmp[i][1] in negative_samples[index:index+20]:
        for i in range(K):
            if(negative_samples[index+i] == dataset_tmp[i][1]):
                negative_sample = negative_samples[index+i]
                while(negative_sample == dataset_tmp[i][1]):
                    negative_sample = np.random.choice(a=words_to_sample, p=probs)
                negative_samples[index+i] = negative_sample


    to_add_list = list(negative_samples[index:index+20])
    for element in to_add_list:
      sample_list.append(element)
    count +=1
    dataset.append(sample_list)

print(dataset[:3])


