# Natural Language Processing

Natural Language processing is a branch of artificial intelligence deals with interaction between machines and humans using Natural Language.

NLP is difficult because, first the words must be understood and then they must be combined together to undertanding the underlying meaning, emotion behind it.

Syntactic analysis and Symantic analysis are the techniques used to complete NLP tasks and these techniques rely on machine learning to derive meaning from human language.

*Syntactic analysis* uses the grammatical rules on the sentence to make sense of it or to get valuable information from it and *Symantic analysis* uses words and how they are interpreted.

We can levearge **Machine Learning** to implement the techniques of Natual Language problem.

#### Machine Learning in NLP

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
