'''
    Created by James Dorfman (github.com/jamesdorfman)
'''
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout, Embedding, TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback
from nltk.probability import FreqDist
import gensim
import string
import re

# CONSTANTS
SEQ_LENGTH = 40
word2vec_dimensions = 100
word2vec_iterations = 1000
word2vec_window = 5
path = 'nas_lyrics.txt'

#STEP 1: LOAD AND CLEAN THE DATA
text = open(path,'r',encoding="utf8").read().lower()

punctuated_lines = text.replace('__seperate__',' ').split('\n')
lines = []
for line in punctuated_lines:
    line = re.sub('['+string.punctuation+']', '', line)
    lines.append(line.split())

songs = text.split('__seperate__')

X = []
y = []

text = ''
for song in songs:
    song = re.sub('['+string.punctuation+']', '', song)
    song = song.replace('\n',' ')
    words = song.split()
    for i in range(len(words)-SEQ_LENGTH-1):
        X.append(' '.join(words[i:i+SEQ_LENGTH]))
        y.append(words[i+SEQ_LENGTH])
    text += ' ' + song

# ENCODE X,y FOR TRAINING
train_X = np.zeros((len(X),SEQ_LENGTH))
train_y = np.zeros(len(X))
for i, seq in enumerate(X):
    for x, word in enumerate(seq.split()):
        train_X[i][x] = word_model.wv.vocab[word].index
    train_y[i] = word_model.wv.vocab[y[i]].index

#STEP 2: TRAIN WORD2VEC MODEL
word2vec_model = gensim.models.Word2Vec(lines, size=Word2Vec_dimensions, min_count=1, window=word2vec_window, iter=word2vec_iterations)
word2vec_weights = word2vec_model.wv.syn0
vocab_size, embedding_layer_size = word2vec_weights.shape

#STEP 3: CREATE AND TRAIN RNN
model = Sequential()
#Add word2vec layer, fill it with embedding weights
model.add(Embedding(input_dim=vocab_size,output_dim=embedding_layer_size,weights=[word2vec_weights]))
#Add LSTM layer so that network can 'remember' previous words
model.add(LSTM(units=embedding_size))
#Expiremental tests found network performed better with a regular 'Dense' layer added here
model.add(Dense(units=vocab_size))
#Softmax layer for one-hot encoding the results
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')

#Callback function to save weights after each epoch
def epoch_callback(epoch_number, _):
    model.save_weights('neural_rap_weights_after_epoch_' + str(epoch_number) + '.h5')

#FIT THE MODEL
model.fit(train_X, train_y,
          batch_size=128,
          epochs=20)
