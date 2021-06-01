# Natural Language Processing

Natural Language processing is a branch of artificial intelligence deals with interaction between machines and humans using Natural Language.

NLP is difficult because, first the words must be understood and then they must be combined together to undertanding the underlying meaning, emotion behind it.

Syntactic analysis and Symantic analysis are the techniques used to complete NLP tasks and these techniques rely on machine learning to derive meaning from human language.

*Syntactic analysis* uses the grammatical rules on the sentence to make sense of it or to get valuable information from it and *Symantic analysis* uses words and how they are interpreted.

We can levearge **Machine Learning** to implement the techniques of Natual Language problem.

## Embeddings

Machine Learning models like numbers and they hate anythinh other than numbers they really do. To get useful information via machine learning, first step is to convert text data into numbers. This post discusses about few methods,

### One-Hot Encoding

Encoding text to numbers. Create a zero vector to the length of the vocabulary(number of unique words on the data) and assign `1` at the index o the word. By doing this what we achieve is called a sparse vector, meaning most indices of the vector are zeros. To form a sentence, concatenate one-hot encoding of the words.

Let's consider a vocabulary with 15,000 words and we encode all of them, what we get is 99.99% of zeros in our data which is really inefficient for training.

### Integer Encoding with unique numbers

Let's switch to use an unique number for each words in the vocabulary. This is efficient thean the above because we'll get a dense vector instead of sparse vector.

But in this method, we lose valuable information to amke something out of the data. Realationship between words is lost. Integer encoding can be challenging for models to interpret, because there is no relationship between similar words and the encodings are alos differnet. This leads to feature-weight combination which is not meaningful.

This where **embedding** comes in

### Word Embeddings

Word embedding is an efficient dense vector way where similar words have similar encodings. Embeddings are floating point vectors, whose values doesn't need to be setup manually. The advantage is the embedding values are learned during training similar to weights of a dense layer. The length of the vector is a parameter to be specified.

The embedding length ranges from 8-dimensional for small datasets to 1024-dimensions for large datasets but requires more data to learn.

[Checkout this notebook for detailed walk through of word embeddings in TensorFlow](https://github.com/JpChii/ML-Projects/blob/main/NLP_word_embeddings.ipynb)

[Visulaization of word embeddings on tensorflow embedding projector](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/JpChii/ML-Projects/main/EmbeddingProjectorconfig.json)

**Natural Language data sources**
Audio, text are sources of Natural Language which are also called as sequence data. Let's see in brief about the neural network suited best for sequential data

**Limitations of fixed input and output shape**
In Vanilla, Convolutional neural network the input and output size has to be fixed. For example in CNN's the input shape is `[height, width]` of the image and a label is predicted for the image. The shape's are fixed, but this is a problem when dealing with sequential data.

Consider this we've 10 sentences with words range from 0 to 25, the input size (i.e) number of words vary across sentences. Now to convert this to a fixed shape,

Fit the input length to a certain number
    * Using a small number, useful information might be lost
    * Using a large number, we might fal into the curse of dimensionality
    
**Recurrent Neural Networks**
    
Still there's a way to use CNN's for Sequence problems but **Recurrent Neural Networks**  are more efficient. We'll check out why that is ?



**Sources:**

1. https://www.tensorflow.org/text/guide/word_embeddings
