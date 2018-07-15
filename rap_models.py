'''
Models for RNN Text generation
'''

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop


def ixs_to_words(list_of_ixs, index_to_word):
    '''
    Produces the sentence represented by the indexes in list_of_ixs
    
    list_of_ixs: list of indexes
    index_to_word: dictionary mapping each index in list_of_ixs to a word
    '''
    return ' '.join([index_to_word[ix] for ix in list_of_ixs])


class Word2VecModel:

    def __init__(self, model = None):
        
        self.model = None
        self.word_to_embedding = {}
        self.word_to_index = {}
        self.index_to_word = {}

        if model:
            self.model = model
            self.extract_w2v_dicts()
    
    def fit(self, sentences, embedding_size = 50, window_size = 5, min_count = 2, num_iterations = 1000):
        
        '''
        Trains a Word2Vec model on sentences
        
        embedding_size: dimensionality of the Word2Vec embeddings
        window_size: maximum distance between context and target words 
        min_count: minimum number of times a word must appear in the vocabulary to have an embedding
        num_iterations: number of epochs the Word2Vec algorithm will run for
        '''

        w2v_model = Word2Vec(
            sentences=sentences,
            size=embedding_size,
            window=window_size,
            min_count=min_count,
            iter=num_iterations
        )

        w2v_model.save('word2vec_weights.model')


        self.model = w2v_model
        self.extract_w2v_dicts()
        self.embedding_size = embedding_size

    def extract_w2v_dicts(self):
        '''
        Creates instance variables for various useful word2vec dictionaries

        Requires: self.model is a valid gensim Word2Vec model
        '''
        word_to_embedding = dict(list(zip(self.model.wv.index2word, self.model.wv.syn0)))
        word_to_index = {word:i for i,word in enumerate(self.model.wv.index2word)}
        index_to_word = {i:word for i,word in enumerate(self.model.wv.index2word)}
        
        # Words that appear < MIN_COUNT times in the vocabulary are given the <UKN> token
        ukn = [0 for i in range(50)] # Assign it the 0 vector
        word_to_embedding['<UKN>'] = ukn
        word_to_index['<UKN>'] = len(word_to_index)
        index_to_word[len(index_to_word)] = '<UKN>'

        self.word_to_embedding = word_to_embedding
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word
        
class RapRNNModel:

    def __init__(self, seq_len, num_targets, w2vModel):
        '''
        w2vModel: A trained Word2VecModel object
        seq_len: Every training example must have this length
        num_targets: Number of possible prediction targets. The Softmax layer will have this many outputs
        '''
        self.seq_len = seq_len

        n_words = len(w2vModel.word_to_embedding)
        embedding_size = w2vModel.embedding_size

        wv_matrix = (np.random.rand(n_words, embedding_size) - 0.5) / 5.0
        for i in range(n_words):
            wv_matrix[i] = w2vModel.word_to_embedding[w2vModel.index_to_word[i]]     
        
        self.w2vModel = w2vModel

        model = Sequential()
        model.add(Embedding(len(wv_matrix), len(wv_matrix[0]), mask_zero=False, weights=[wv_matrix], input_length=seq_len, trainable=True))
        model.add(LSTM(100,return_sequences=False))
        model.add(Dense(num_targets, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Model Summary:")
        print(model.summary())

        trained_model = model
        try:
            trained_model.load_weights('model_weights.h5')
            model = trained_model
        except Exception as e:
            print('No pre-trained weights found')

        self.rnn_model = model


    def fit(self, X, y, batch_size = 128, epochs = 50):
        '''
        Trains the model on the labelled set of sequences (X, y)

        X: Numpy array of shape (num_training_examples, max_seq_len) 
           Each entry is an index, where 
           If a sequence has a length less then max_seq_len, then pad it with zeros
        y: Numpy array of shape (num_training_examples, vocabulary_size)
               Each entry is one-hot encoded
        '''

        for epoch in range(1, epochs + 1):
            print('Training epoch', epoch)
            print('-' * 50)
            self.rnn_model.fit(X, y, batch_size=128, nb_epoch=2)
            self.rnn_model.save_weights('model_weights.h5', overwrite=True)
            print('Sampled text from rap language model:')
            print(self.sample())

        print("Finished training model on",epochs,"epochs")

    def sample(self, sample_length = 30):
        '''
        Produces a rap line of length max_len using model
        
        word2VecModel: a Word2VecModel that has been fit() on a set of sentences
        '''
        
        # Initialize sample with the zero vector
        word_idxs = np.zeros((1,self.seq_len))
        
        # Sample each character, feed new input back into network
        for i in range(self.seq_len):
            sample_prob = self.rnn_model.predict(word_idxs)
            idx = np.random.choice(len(sample_prob[0,:]), p=sample_prob[0,:])
            word_idxs[0,i] = idx
            
        return ixs_to_words(word_idxs[0,:], self.w2vModel.index_to_word)
