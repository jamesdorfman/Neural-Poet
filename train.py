#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Train the RNN poetry language model
'''

import re
import string
import sys
import pickle

from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from tensorflow import logging

from constants import *
from models import PoetryRNNModel, Word2VecModel

__author__ = "James Dorfman (github.com/jamesdorfman)"
__copyright__ = "Copyright 2018, James Dorfman"
__license__ = "GNU"

logging.set_verbosity(logging.ERROR)

WORDS_BETWEEN_SEQUENCES = 1
# Number of training epochs
try:
    EPOCHS = int(sys.argv[1])
except Exception as e:
    EPOCHS = 100

# Get poems
text = open('res/poems.txt', 'r', encoding="utf8").read().lower()
poems = [ t.replace('\n', ' <NEWLINE> ') + ' <END>' for t in text.split('\n\n') ]
frequent_words = set([k for k, v in Counter(text).items() if v >= 3]) # Words not in here will be encoded with <UKN>

pretrained_vectors = open('res/word2vec.50d.filtered_for_poetry.txt', 'r', encoding='utf8')
w2vModel = Word2VecModel(embedding_size=50)

for line in pretrained_vectors:
    if len(line) > 0:
        word = line.split()[0]
        embedding = line.split()[1:]
        w2vModel.add(word, embedding)

# The custom tokens in poems.txt are not present in the pretrained embeddings
# Create vectors for them and add them to the model
w2vModel.add('<UKN>', [0 for i in range(50)]) # Assign <UKN> the 0 vector
w2vModel.add('<END>', [10 for _ in range(50)]) # <END> can be assigned any arbitrary vector, since it it only every terminates sequences
w2vModel.add('<NEWLINE>', [5 for _ in range(50)]) # We don't have a pretrained vector for <NEWLINE>, so we will assign it an arbitrary one and rely on backpropogation to tune it (since the embedding layer's weights are trainable)

# Convert all poems (lists of words) to lists of indices
poems_ix = []
for poem in poems:
    cur_ix = []
    for word in poem.split():
        if word in w2vModel.word_to_index:
            cur_ix.append(w2vModel.word_to_index[word])
        else:
            cur_ix.append(w2vModel.word_to_index['<UKN>'])
    poems_ix.append(cur_ix)

# Create training examples from each poem
# Each training example is the (X, y) pair (30 word sequence from poem, 31st word of sequence)
# If a sequence is less than 30 words long, it is padded with 0s
X = []
y = []
for poem in poems_ix:
    i = 0
    while i < len(poem):
        start = max(i - SEQ_LEN, 0)
        X.append(poem[start:i])
        y.append(poem[i])
        i += WORDS_BETWEEN_SEQUENCES
X = pad_sequences(X, maxlen=SEQ_LEN, padding='post')
y_onehot = np.array(pd.get_dummies(y))

print('\nTRAINING LANGUAGE MODEL')
print('-' * 50)

rnn = PoetryRNNModel(SEQ_LEN, len(list(set(y))), w2vModel)

# Pickle and save the PoetryRNNModel object (model architecture + w2vModel)
# This lets us  access it when making predictions, in `predict.py`
with open(RNN_PICKLE_FILE, 'wb') as pickle_output:
    pickle.dump(rnn, pickle_output, pickle.HIGHEST_PROTOCOL)

rnn.init_model() # Must be called after pickling the object - embedding layers can't be pickled

rnn.fit(X, y_onehot, epochs=EPOCHS)
