'''
Created by James Dorfman (github.com/jamesdorfman)
'''

import string
from rap_models import Word2VecModel
from rap_models import RapRNNModel
import re

import numpy as np
import pandas as pd

from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

# Get lyrics
file_path = 'nas_lyrics.txt'
text = open(file_path, 'r', encoding="utf8").read().lower()
text = text.replace('<seperate>',' ') # In the text file, all songs are seperated by <seperate>

# Remove punctuation from songs
lines = text.split('\n')
lines = [re.sub(r'[^\w\s]', '', line) for line in lines]

filtered_lines = []
for line in lines:
    # We don't want lines that are too short (meaningless) or too big ()
    if len(line.split()) >= 10 and len(line.split()) < 40:
        filtered_lines.append(line)
split_lines = [line.split() for line in filtered_lines]

max_len = 0
for line in split_lines:
    if len(line) > max_len:
        max_len = len(line)

print('training Word2Vec model...')
w2vModel = Word2VecModel()
w2vModel.fit(split_lines)

most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in w2vModel.model.most_similar("maybe")[:8])
print('  %s -> %s' % ("maybe", most_similar))

lines_ix = []
for line in split_lines:
    cur_ix = []
    for word in line:
        if word in w2vModel.word_to_index.keys():
            cur_ix.append(w2vModel.word_to_index[word])
        else:
            cur_ix.append(w2vModel.word_to_index['<UKN>'])
    lines_ix.append(cur_ix)

X = []
y = []
for line in lines_ix:
    for i in range(1,len(line)):
        X.append(line[:i])
        y.append(line[i])
X = pad_sequences(X, maxlen=max_len, padding='post')
y_onehot = np.array(pd.get_dummies(y))

rnn = RapRNNModel(max_len, len(list(set(y))), w2vModel)
rnn.fit(X, y_onehot)