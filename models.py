'''
Models for text generation
'''

import string
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import regularizers
from keras.layers import Dense, Embedding, LSTM, CuDNNLSTM, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import adam
from tensorflow.python.client import device_lib

__author__ = "James Dorfman"
__copyright__ = "Copyright 2017-present"
__license__ = "GNU"
__version__ = "1.2"

import tensorflow as tf
class Word2VecModel:

    def __init__(self, model = None, embedding_size=50):
        
        self.model = None
        self.word_to_embedding = {}
        self.word_to_index = {}
        self.index_to_word = {}
        self.embedding_size = embedding_size

        if model:
            self.model = model
            self.extract_w2v_dicts()
    
    def fit(self, sentences, window_size = 5, min_count = 2, num_iterations = 1000):
        
        '''
        Trains a Word2Vec model on sentences
        
        embedding_size: dimensionality of the Word2Vec embeddings
        window_size: maximum distance between context and target words 
        min_count: minimum number of times a word must appear in the vocabulary to have an embedding
        num_iterations: number of epochs the Word2Vec algorithm will run for
        '''

        w2v_model = Word2Vec(
            sentences=sentences,
            size=self.embedding_size,
            window=window_size,
            min_count=min_count,
            iter=num_iterations
        )

        w2v_model.save('res/word2vec_weights.model')

        self.model = w2v_model
        self.extract_w2v_dicts()
        self.embedding_size = embedding_size

        self.word_to_embedding = dict(list(zip(self.model.wv.index2word, self.model.wv.syn0)))
        self.word_to_index = {word:i for i,word in enumerate(self.model.wv.index2word)}
        self.index_to_word = {i:word for i,word in enumerate(self.model.wv.index2word)}
        
        # Words that appear < MIN_COUNT times in the vocabulary are given the <UKN> token
        self.add('<UKN>', [0 for _ in range(50)]) # Assign it to the 0 vector

    def add(self, word, embedding):
        self.word_to_embedding[word] = embedding
        self.word_to_index[word] = len(self.word_to_index)
        self.index_to_word[len(self.index_to_word)] = word

    def ixs_to_words(self, list_of_ixs):
        '''
        Produces the sentence represented by the indexes in list_of_ixs
        
        list_of_ixs: list of indexes
        index_to_word: dictionary mapping each index in list_of_ixs to a word
        '''
        # Put spaces between all words
        sentence = ' '.join([self.index_to_word[ix] for ix in list_of_ixs])

        # Remove spaces between letters and punctuation (ex. I came , I saw -> I came, I saw)
        filtered_sentence = ''
        i = 1
        while i < len(sentence):
            if not (sentence[i] in string.punctuation and sentence[i-1] == ' ') or sentence[i] in ['<','>']:
                filtered_sentence += sentence[i-1]
            i += 1
        filtered_sentence += sentence[len(sentence)-1]
        
        return filtered_sentence.replace('<NEWLINE> ','\n').replace('<END>','')
        
class PoetryRNNModel:

    '''
    This class can train/sample a pure Tensorflow model or a Keras Tensorflow model:
        
    The pure TensorFlow model was built to replace the Keras model as it is more customizable.
    Although the TensorFlow model works just as well as the Keras model, it is significantly slower to train on CPU
    This is because the Keras API is optimized very well.
    I am working on further optimizing the TensorFlow code so that it can replace the Keras model.
    '''

    def __init__(self, seq_len, num_targets, w2vModel, batch_size=128, model_api='keras'):
        '''
        w2vModel: A trained Word2VecModel object
        seq_len: Every training example must have this length
        num_targets: Number of possible prediction targets. The Softmax layer will have this many outputs
        model_api: One of ['keras', 'tensorflow'], determines the API used to construct the actual RNN model

        NOTE: Call self.init_model to initalize the actual RNN model.
              Models with embedding layers cause IOErrors when pickled, so moving this functionality
              to a dedicated method enables pickling of class instances.
        '''


        self.seq_len = seq_len
        self.num_targets = num_targets
        self.w2vModel = w2vModel
        self.batch_size = batch_size
        if model_api == 'keras':
            self.keras = True
        else:
            self.keras = False

    def init_model(self):
        self.init_embedding_matrix()
        if self.keras:
            self.init_keras_model()
        else:
            self.init_tensorflow_model()

    def init_embedding_matrix(self):
        '''
        Initializes a matrix with word embeddings for each word in the corpus.
        This makes model training more efficient, since each word in a batch 
        takes up store 1/embedding_size as much space in RAM
        
        Matrix is stored in self.wv_matrix
        '''
        self.n_words = len(self.w2vModel.word_to_embedding)
        self.embedding_size = self.w2vModel.embedding_size

        self.wv_matrix = (np.random.rand(self.n_words, self.embedding_size) - 0.5) / 5.0 # Randomly initialize matrix, in case a word doesn't have a pre-trained embedding
        for i in range(self.n_words):
            embedding = self.w2vModel.word_to_embedding[self.w2vModel.index_to_word[i]]
            if embedding is not None:
                self.wv_matrix[i] = embedding

    def init_keras_model(self):
        '''
        Creates the model 
            embedding layer -> LSTM layer -> Softmax layer
        This model is then assigned to self.model
        '''  

        model = Sequential()
        model.add(Embedding(len(self.wv_matrix), len(self.wv_matrix[0]), mask_zero=False, weights=[self.wv_matrix], input_length=self.seq_len, trainable=True))

        use_gpu = False
        for elem in device_lib.list_local_devices():
            if elem.device_type == 'GPU':
                use_gpu = True
        if use_gpu:
            model.add(CuDNNLSTM(100, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), return_sequences=False))
        else:
            model.add(LSTM(100, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), return_sequences=False))
        model.add(Dense(self.num_targets, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        try:
            trained_model = model
            model_weights_file = 'res/model_weights.h5'
            trained_model.load_weights(model_weights_file)
            model = trained_model
            print('Using pretrained weights from the file ',model_weights_file)
        except Exception as e:
            print('No pre-trained weights found')

        self.rnn_model = model

    def init_tensorflow_model(self):
        '''
        Creates the model 
            embedding layer -> LSTM layer -> Softmax layer
        This model is then assigned to self.model
        '''

        inputs = tf.placeholder(tf.int32, [None,None])
        labels = tf.placeholder(tf.int32, [None, None], name='labels')

        embedding = tf.get_variable("embedding_layer", initializer=wv_matrix)
        embedding = tf.cast(embedding, tf.float32)
        embed = tf.nn.embedding_lookup(embedding, inputs)

        use_gpu = False
        for elem in device_lib.list_local_devices():
            if elem.device_type == 'GPU':
                use_gpu = True
        if use_gpu:
            cell =  tf.contrib.cudnn_rnn.CuDNNLSTM(100)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)#, #initial_state=initial_state)
        else:
            cell = tf.contrib.rnn.LSTMBlockFusedCell(100)
            lstm_outputs, final_state = cell(embed, dtype=tf.float32)#, #initial_state=initial_state)

        lstm_outputs = lstm_outputs[:,:,-1] # We only want the output from the very last LSTM (we are not trying to output a sequence)

        # Make a layer to process LSTM layer (make network architecture LSTM -> DENSE -> SOFTMAX)
        W = tf.Variable(tf.random_normal([self.embedding_size, self.n_words]))
        b = tf.Variable(tf.random_normal([self.n_words]))

        #lstm_outputs = tf.reshape(lstm_outputs, [-1, num_hidden])
        logits = tf.matmul(lstm_outputs, W) + b

        prediction = tf.nn.softmax(logits)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                              logits=logits, labels=labels))

        optimizer = tf.train.AdamOptimizer(0.01)
        train_step = optimizer.minimize(loss)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Store TensorFlow variables so that we can run the graph when training / predicting
        self.inputs = inputs
        self.labels = labels
        self.loss = loss
        self.final_state = final_state
        self.train_step = train_step
        self.prediction = prediction
        self.accuracy = accuracy

    def fit(self, X, y, epochs = 50):
        '''
        Trains the model on the labelled set of sequences (X, y)

        X: Numpy array of shape (num_training_examples, max_seq_len) 
           Each entry is an index, where 
           If a sequence has a length less then max_seq_len, then pad it with zeros
        y: Numpy array of shape (num_training_examples, vocabulary_size)
               Each entry is one-hot encoded
        '''
        epochs_between_samples = 5
        epochs_so_far = 0
        while epochs_so_far < epochs:

            # Ensure we don't do too many epochs if epochs % epochs_between_samples != 0            
            intermediate_epochs = min(epochs_between_samples, epochs - epochs_so_far)
            
            print('\nTRAINING EPOCHS ', epochs_so_far + 1, '-',epochs_so_far + intermediate_epochs)
            print('-' * 50)
            self.fit_model(X, y, epochs=intermediate_epochs, epoch_start=epochs_so_far+1)
            print('Language model sample:')
            print(self.sample())
            
            epochs_so_far += intermediate_epochs

        print("Finished training model on",epochs,"epochs")

    def get_batches(self, X, y):
        '''
        Seperates X and y into minibatches of size `self.batch_size` 
        Trailing elements that don't fit into a full batch are truncated
        
        Produces the a list of minibatche tuples (minibatch_X, minibatch_y)
        '''

        batches = []
        n_batches = len(X)//self.batch_size
        batched_X, batched_y = X[:n_batches*self.batch_size], y[:n_batches*self.batch_size] # We can't have batches with # training examples < batch_size
        num_batches = len(batched_X)
        for i in range(0, num_batches, self.batch_size):
            batches.append([batched_X[i:i+self.batch_size], batched_y[i:i+self.batch_size]])
        return batches

    def fit_model(self, X, y, epochs=100, epoch_start=1):
        '''
        Fits the Recurrent Neural Network to (X, y)
        epoch_start: Where to start counting epochs; used for printing epoch # after each epoch
        '''

        if self.keras:
            self.rnn_model.fit(X, y, batch_size=self.batch_size, epochs=epochs)
            self.rnn_model.save_weights('res/model_weights.h5', overwrite=True)
        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                n_batches = len(batches)
                for epoch in range(epochs):
                    start_time = time.time()

                    train_acc = []
                    train_loss = []
                    for batch_num, (minibatch_x, minibatch_y) in enumerate(batches):
                        feed = {self.inputs: minibatch_x,
                                self.labels: minibatch_y}
                        batch_loss, self.state, _, batch_acc = sess.run([self.loss, self.final_state, self.train_step, self.accuracy], feed_dict=feed)
                        train_acc.append(batch_acc)
                        train_loss.append(batch_loss)                
                    
                    end_time = time.time()

                    print("Finished epoch {epoch} in {time}s: loss={loss}, accuracy={accuracy}".format(
                        time="{0:.3g}".format(end_time-start_time),
                        epoch=epoch+epoch_start,
                        loss="{0:.4g}".format(np.mean(batch_loss)),
                        accuracy="{0:.4g}".format(np.mean(train_acc))))        

    def sample(self, sample_length = 30):
        '''
        Produces a rap line of length max_len using model
        
        word2VecModel: a Word2VecModel that has been fit() on a set of sentences
        '''

        words = []
        
        # Boolean to track if previous word was punctuation
        # Used to prevent picking 2 pieces of punctuation in a row
        prev_punctuation = False 

        # Sample the next word, feed new input back into network
        for i in range(200): # 200 words or <END>, whichever comes first
            padded_words = words[len(words)-self.seq_len:]
            while len(padded_words) < self.seq_len: # Pad rest of sequence with 0s
                padded_words.append(0)
            
            if self.keras:
                word_idxs = np.array(padded_words).reshape(1, self.seq_len) # model.predict() expects a Numpy array of shape (1, SEQ_LEN)
                sample_prob = self.rnn_model.predict(word_idxs)
            else:
                 with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    word_idxs = np.array(padded_words)#.reshape(1, self.seq_len) # model.predict() expects a Numpy array of shape (1, SEQ_LEN)
                    feed = {self.inputs: [word_idxs for _ in range(self.batch_size)]}
                    sample_prob = sess.run(self.prediction, feed) # [0] !!!

            idx = self.w2vModel.word_to_index['<UKN>']
            # Keep picking words until we get one that's not <UKN>
            # If the previous word was punctuation, keep picking words until we don't get punctuation
            while idx == self.w2vModel.word_to_index['<UKN>'] or (prev_punctuation and self.w2vModel.index_to_word[idx] in string.punctuation):
                idx = np.random.choice(len(sample_prob[0,:]), p=sample_prob[0,:])
            words.append(idx)

            if self.w2vModel.index_to_word[idx] in string.punctuation:
                prev_punctuation = True

            if self.w2vModel.index_to_word[idx] == '<END>':
                break
            
        return self.w2vModel.ixs_to_words(words)
