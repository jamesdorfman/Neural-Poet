# Neural Rap
A Word-level Recurrent Neural Network trained on the discography of Nas

Sample rap lines produced by the network:

` The way I poisoned you to throw out rules of rap`

` I'm the only one that carries that vision`

### Project setup
Use a package manager like `pip` to install the following dependencies:
* `Numpy`<br />
* `Pandas`<br />
* `Tensorflow`<br />
* `Keras`<br />
* `Gensim`<br />

Then, run `predict.py` to sample the rap language model. Run `train.py` first if you would like to train the RNN's weights.

# FAQ

### Why does the network unable to maintain a coherent stream of thought?
Nas is famous for his intricate and complex rhyme schemes. In fact, his rhymes  have even been studied formally in academia. This wide ranging vocabulary makes it difficult for even complex models like RNNs to properly understand the long-range dependencies in his sentances.
Furthmore, this network was only trained for a short amount of time , since compute time is expensive!

### Why use an RNN?
However, sometimes such simple models aren't good enough. There are many complex features of the data that are intricately connected and the models are unable to find relationships

A neural network is essentially just specific model which is able to learn very complex relationships in data, often even creating its own features for the data!

### What is the purpose of an embedding layer?
The corpus used to train the model consisted of over 20k words. Storing the individual Word2Vec embeddings for each word requires a lot of storage space and slows down training.
The embedding layer (the first layer in the network) takes in a word ID and outputs that word's Word2Vec embedding. This largly reduces the size of the training set, as it allows each high-dimensional vector to be replaced with a single integer ID.

# Acknowledgements
This project was inspired by Andrej Karparthy's infamous blog post: http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
Andrej trained a recurrent neural network to read in text character-by-character, and predict the next character that would appear. I decided to try to predict text word-by-word, and incoporating Word2Vec embeddings, in order ot produce more coherent sentances
