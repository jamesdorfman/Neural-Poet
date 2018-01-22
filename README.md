# NeuralRap
Rap from an LSTM Recurrent Neural Network trained on the discography of Nas

### Project setup
Make sure you have keras, numpy and gensim installed.
Then, either run train.py to generate new weights, or use the weights which were pretrained on a Google Cloud Compute VM for one hour.
Then, run predict.py

# FAQ

### Why does the network have trouble maintaining coherent thought?
Nas' rhymes have been studied in academia because he displays a level of verbal skill rarely seen in rap. He has a wide ranging vocabulary which makes extremely difficult for the network to learn his style. Furthmore, this network was trained using Google's Cloud Compute Servers. More training requires more server time, and server time is expensive!

### What inspired this project?
This project was inspired by Andrej Karparthy's infamous blog post: http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 
Andrej trained a recurrent neural network to read in text character-by-character, and predict the next character that would appear. I decided to try to predict text word-by-word to see if I could get better results

### What is a neural network?
Machine Learning is all about creating models to predict things. Often, simple models will suffice. A well-known example of this is linear regression, which trains a model to predict an unknown class, such as the price of a house, from known values like the number of rooms and the size of the backyard. This model is very popular and simple to understand.
However, sometimes such simple models aren't good enough. There are many complex features of the data that are intricately connected and the models are unable to find relationships

A neural network is essentially just specific model which is able to learn very complex relationships in data, often even creating its own features for the data!

### What problems did you face during this project?
The biggest problem I faced was the size of the corpus used to train the network. When predicting text character-by-character, the network has much less possiblities for the next character, then when predicting text word-by-word, as this specific corpus had ~20,000 unique words

### How did you solve these problems?
I originally tried using a popular technique called one-hot-encoding to categorize each word. However, this involved creating a vector with 20,000 numbers for each word, with 19,999 of these numbers being 0, and one of them being 1. This was very inefficient, and my computer had trouble storing such a large data set I ended up encoding each word with its own integer, 
So instead of inputting words as numbers and trying to get the network to learn their relationships, I first used the Gensim library to create an 100 dimensional vector for each word. These vectors are special because, unlike one-hot-encoding which doesn't contain any relational information about the word, these Word2Vec vectors contain lots of relational information. In fact, they even allow for cool operations like substracting two words from eachother, and getting an answer. For example, my model said that king - girl = queen !

The benefit of this model was that not only did it shrink the size of the input to the model, but by creating an embedding layer in the network containing these 100-Dimensional vectors, I was able to store each word as an individual integer which was decoded into its Word2Vec vector after going through the embedding layer. This allowed me to transform each sequence of text into a list of numbers, which largly reduced the size of the dataset. Furthermore, the relationsal aspect of the Word2Vec model yielded better results.

