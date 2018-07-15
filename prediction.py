'''
Created by James Dorfman (github.com/jamesdorfman)
'''

import rap_models

from gensim import Word2Vec

pretrained_weights_path = 'word2vec_weights.model'
pretrained_w2v_model = Word2Vec.load_weights(pretrained_weights_path)

w2vModel = Word2VecModel(pretrained_w2v_model)
rnn = RapRNNModel(w2v_model)
print(w2v_model.sample(30))
