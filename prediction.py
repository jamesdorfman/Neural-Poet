'''
    Created by James Dorfman (github.com/jamesdorfman)
'''
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import numpy as np
from keras.layers import Embedding
import gensim
import string
import re

path = 'nas_lyrics.txt'

#LOAD AND CLEAN THE DATA
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

#LOAD PRETRAINED WEIGHTS
word_model = gensim.models.Word2Vec.load('trained_word2vec')
word2vec_weights = word2vec_model.wv.syn0
vocab_size, embedding_layer_size = word2vec_weights.shape

def predict(phrase,num_to_generate):
    phrase = phrase.lower()
    words = phrase.split()
    sequenced_phrase = []
    for word in words:
        sequenced_phrase.append(word_model.wv.vocab[word].index
    X = np.array(sequenced_phrase)
    for i in range(num_to_generate):
        sequenced_phrase.append(model.predict(x=X))
    generated = ''
    for id in sequenced_phrase:
        id_to_word = word_model.wv.index2word[id]
        generated.join(id_to_word)
    return generated

#TODO: Add a degree of randomness for different words every time

#RECREATE MODEL, explanation can be found in predict.py
model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embedding_layer_size,weights=[word2vec_weights]))
model.add(LSTM(units=embedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')

#LOAD PRESAVED WEIGHTS
model.load_weights('neural_rap_weights.h5')

#DO PREDICTION
print(predict(phrase,15))
