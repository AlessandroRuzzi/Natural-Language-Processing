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
from numpy import save
from matplotlib import pyplot as plt

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


W = np.load("W.npy")

from numpy.linalg import norm


def cosine_similarity(emd_x,emd_y):  # function to calculate similarity between two words
  return np.round(np.dot(emd_x,emd_y)/(norm(emd_x)*norm(emd_y)),decimals=4)


x = ["film","film","home","home","father","father","street","writer","writer","boy","children","children","eat","eat","water","water"]
y = ["movie","water","house","yellow","mother","street","avenue","poet","potatoes","girl","young","old","food","sport","liquid","solid"]

for i in range(len(x)):
    sim = cosine_similarity(W[word_to_index[x[i]]],W[word_to_index[y[i]]])
    print(f"Similarity between {x[i]} and {y[i]} is : {sim}")


x = ["love", "car", "president", "monday", "green", "money",
     "health", "faith", "book","france", "swiss", "spring",
     ##my words
     "castle", "phone", "sand", "toilet", "sea",
     ]

for word in x:
    cosine = []
    for i in range(W.shape[0]):
      cosine.append(cosine_similarity(W[i],W[word_to_index[word]]))
    index = np.argsort(cosine)
    to_print = []
    for i in range(1,11):
      to_print.append((index_to_word[index[W.shape[0] -1 - i]], cosine[index[W.shape[0] -1 -i]]))
    print(word + " -> " + str(to_print))


x_1 = ["london", "father", "children", "sister", "happiness", "light", "food", "friend"]
x_2 = ["england", "man", "young", "boy", "good", "day", "eat", "love"]
x_3 = ["germany", "woman", "old", "boy", "bad", "night", "drink", "hate"]

y = ["berlin", "mother", "parents", "brother", "pain", "darkness", "drink", "enemy"]

for i in range(len(y)):
  sim = cosine_similarity(W[word_to_index[x_1[i]]] - W[word_to_index[x_2[i]]] + W[word_to_index[x_3[i]]],
                          W[word_to_index[y[i]]])
  print(f"Similarity between [{x_1[i]} - {x_2[i]} + {x_3[i]}]and {y[i]} is : {sim}")
"""
from sklearn.manifold import TSNE

labels = []
tokens = []

for i in range(len(index_to_word)):
    tokens.append(W[i,:])
    labels.append(index_to_word[i])

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

save("new_values.npy",new_values)
"""

new_values = np.load("new_values.npy")

from sklearn.cluster import AgglomerativeClustering
from adjustText import adjust_text

np.random.seed(42)

plt.style.use('default')  # use default values to plot the embeddings


def plot_words(new_values, labels, to_plot, title, prob=0.0, limit=100, dim1=15, dim2=10):
    visualization_dataset = np.array(new_values[:to_plot])
    plt.rcParams["figure.figsize"] = (dim1, dim2)

    to_adjust = []
    for i in np.unique(labels):
        plt.scatter(visualization_dataset[np.array(labels)[:to_plot] == i, 0],
                    visualization_dataset[np.array(labels)[:to_plot] == i, 1], label=i,
                    s=20)  # plot the embedding points
        tmp_labels = np.array(labels == i)
        indices = []

        for index in range(to_plot):
            if tmp_labels[index]:
                indices.append(index)
        for j in range(len((visualization_dataset[np.array(labels)[:to_plot] == i, 0]))):
            if (len(indices) < limit):  # don't plot texts for bigger clusters
                if (np.random.rand() > prob):  # print only a portion of the words to improve readibility
                    to_adjust.append(plt.text((visualization_dataset[np.array(labels)[:to_plot] == i, 0])[j],
                                              (visualization_dataset[np.array(labels)[:to_plot] == i, 1])[j],
                                              index_to_word[
                                                  indices[j]]))  # plot the word on the correspnding embedding point

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    adjust_text(to_adjust)
    plt.show()


ag = AgglomerativeClustering(n_clusters=8).fit(
    new_values)  # use the fit function of the Clustering method using all the samples
labels = ag.labels_

plot_words(new_values, labels, 100, 'Embedding plot 100 samples', )

plot_words(new_values, labels, 500, 'Embedding plot 500 samples', 0.40, 150, dim1=20, dim2=15)

plot_words(new_values, labels, 1000, 'Embedding plot 1000 samples', 0.7, 250, dim1=30, dim2=20)


