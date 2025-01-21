# Basics of NLP

In this notebook we're gonna recap basics of NLP. The topics include:

1. Bag of words
2. Stemming
3. N-grams
4. Tf-Idf
5. Word2Vec

We'll use Amazon Fine Food Reviews Dataset, and problem of sentiment analysis to understand these concepts.

Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Data includes(from kaggle):

Reviews from Oct 1999 - Oct 2012
* 568,454 reviews
* 256,059 users
* 74,258 products
* 260 users with > 50 reviews

Our target variable is Score in the dataset(review ratings):
1. When score > 3 Positive
2. When score < 3 Negative
3. When Score = 3 Remove this from dataset.



```python
# imports
%matplotlib inline

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from nltk.stem.porter import PorterStemmer
```

## EDA


```python
# We'll use the sqlite file from dataset to load it into pandas
# We'll remove score == 3 here
conn = sqlite3.connect("data/database.sqlite")

# Read in the data
filtered_data = pd.read_sql_query(
    sql="""
Select * from Reviews where Score != 3
""",
con=conn,
)
```


```python
filtered_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 525814 entries, 0 to 525813
    Data columns (total 10 columns):
     #   Column                  Non-Null Count   Dtype 
    ---  ------                  --------------   ----- 
     0   Id                      525814 non-null  int64 
     1   ProductId               525814 non-null  object
     2   UserId                  525814 non-null  object
     3   ProfileName             525814 non-null  object
     4   HelpfulnessNumerator    525814 non-null  int64 
     5   HelpfulnessDenominator  525814 non-null  int64 
     6   Score                   525814 non-null  int64 
     7   Time                    525814 non-null  int64 
     8   Summary                 525814 non-null  object
     9   Text                    525814 non-null  object
    dtypes: int64(5), object(5)
    memory usage: 40.1+ MB



```python
filtered_data.Score.unique()
```




    array([5, 1, 4, 2])




```python
# convert Score from ratings to postive and negative
def partition(x):
    if x > 3:
        return "positive"
    return "negative"

actualScore = filtered_data["Score"]
postiveNegative = actualScore.map(partition)

# Set the score variale in df to this
filtered_data["Score"] = postiveNegative
```


```python
filtered_data["Score"].unique()
```




    array(['positive', 'negative'], dtype=object)



## Data Cleaning Depluication

Grabage in --> Garbage out

There are duplicates in review for a single product's different flavours. Like a wafer cookies with differnet flavours. This is not useful as the data is duplicate and overfit the model or while splitting the data. Let's see a sample of this duplicate and remove them. This process is called deduplication.



```python
sample_duplicates = pd.read_sql_query(
    "select * from reviews where UserId = 'AR5J8UI46CURR'",
    conn,
)
sample_duplicates
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>73791</td>
      <td>B000HDOPZG</td>
      <td>AR5J8UI46CURR</td>
      <td>Geetha Krishnan</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1199577600</td>
      <td>LOACKER QUADRATINI VANILLA WAFERS</td>
      <td>DELICIOUS WAFERS. I FIND THAT EUROPEAN WAFERS ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78445</td>
      <td>B000HDL1RQ</td>
      <td>AR5J8UI46CURR</td>
      <td>Geetha Krishnan</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1199577600</td>
      <td>LOACKER QUADRATINI VANILLA WAFERS</td>
      <td>DELICIOUS WAFERS. I FIND THAT EUROPEAN WAFERS ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>138277</td>
      <td>B000HDOPYM</td>
      <td>AR5J8UI46CURR</td>
      <td>Geetha Krishnan</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1199577600</td>
      <td>LOACKER QUADRATINI VANILLA WAFERS</td>
      <td>DELICIOUS WAFERS. I FIND THAT EUROPEAN WAFERS ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>138317</td>
      <td>B000HDOPYC</td>
      <td>AR5J8UI46CURR</td>
      <td>Geetha Krishnan</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1199577600</td>
      <td>LOACKER QUADRATINI VANILLA WAFERS</td>
      <td>DELICIOUS WAFERS. I FIND THAT EUROPEAN WAFERS ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>155049</td>
      <td>B000PAQ75C</td>
      <td>AR5J8UI46CURR</td>
      <td>Geetha Krishnan</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>1199577600</td>
      <td>LOACKER QUADRATINI VANILLA WAFERS</td>
      <td>DELICIOUS WAFERS. I FIND THAT EUROPEAN WAFERS ...</td>
    </tr>
  </tbody>
</table>
</div>



We can see the context exaplined above code cell in these three datapoints, tiestamp, score etc are all the same. If you checkout the ProductId(ASIN) you'll notice they're different flavours of the same product. We'll remove the duplicates by keeping only a single review of a product by a single user.

We'll sort the data and use drop_duplicates to perform this operation.


```python
# SOrting the data with productId
sorted_data = filtered_data.sort_values(
    by="ProductId",
    axis=0, # by Columns
    ascending=True,
)
```


```python
# Dedpuplication
final = sorted_data.drop_duplicates(
    subset={
        "UserId",
        "ProfileName",
        "Time",
        "Text",
    }
)
```


```python
final.shape
```




    (364173, 10)



Initally there were 500k+ data points now it's reduced to 360k+.

Denominator should always be greater than numerator, let's check if there's any samples.


```python
final[final["HelpfulnessNumerator"] > final["HelpfulnessDenominator"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59301</th>
      <td>64422</td>
      <td>B000MIDROQ</td>
      <td>A161DK06JJMCYF</td>
      <td>J. E. Stephens "Jeanne"</td>
      <td>3</td>
      <td>1</td>
      <td>positive</td>
      <td>1224892800</td>
      <td>Bought This for My Son at College</td>
      <td>My son loves spaghetti so I didn't hesitate or...</td>
    </tr>
    <tr>
      <th>41159</th>
      <td>44737</td>
      <td>B001EQ55RW</td>
      <td>A2V0I904FH7ABY</td>
      <td>Ram</td>
      <td>3</td>
      <td>2</td>
      <td>positive</td>
      <td>1212883200</td>
      <td>Pure cocoa taste with crunchy almonds inside</td>
      <td>It was almost a 'love at first bite' - the per...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's remove these
final = final[final["HelpfulnessNumerator"] <= final["HelpfulnessDenominator"]]
final.shape
```




    (364171, 10)




```python
# Let's check the class difference
final.Score.value_counts()
```




    Score
    positive    307061
    negative     57110
    Name: count, dtype: int64



Positive classes are 6X of Negative classes. This is an imbalanced dataset.

## Why Text to Vector?

The two main feature for our problem is Summary and Text. But these feature are in Natural Language and ML models don't understand anything other than numbers. If we can convert these text to vectors, we can leverage linear algebra techniques we've learnt to classify the data points.

We can cluster the datapoints and seperate them in d dimesnsional space of vectors.

Thinking interms of linear algebra, if the two classes are seperable by a plane. We can use the equation of hyperplane with $W_T$ vector. If the point and $W_T$ vector are on the same side it's positive and negative vice versa.

Intutions Behind this:
1. Semantically similar text must be closer geometrically
2. Semantic similarity is inversley proportion to distance.

But how do we convert text to vectors?
1. BOW
2. TF-IDF
3. Word2Vec
4. Avg-Word2Vec
5. TfIDF-Word2Vec

## Bag of Words

To create a Bag of Words(BOW):

1. Create a set of all unique words in the corpus(All Documents or reviews in our ue case)
2. Each word in the vocab forms a dimension
3. We'll create a d-dimension vector(where d represents each unique word)

**Characterisitcs of Bag of Words Vector:**

To create a vector for a document - Fill the d-dimension vector with 1 for words present in document and 0 for words not in document. This is a binary BoW vector. These binary BoW vectors are sparse(has lot's of zeros). Why?

Assume the 500k+ plus review, imagine the number of unique words across these reviews in it's entirety vs number of words in a single review. We'll end up with lots of zeros in the vector. Hence Sparse.

Imageine D-Dimension vectors as row vector for filling it up.

There's another type of BoW called count BoW where will fill the vector with number of times a word has occured instead of filling with 0's and 1's.

**Problems with this approach:**

* While we calculate distance between tow BoW's is just norm of ||v1-v2||, this is equivalent to the words not matching in two sentences. If only not is the differnce between two sentence, the distance will be small but it should be large due to the negation(not). The objective of achieving semantic similarity is lost here.
* Semantic similarity between words is also lost(tasty and delecious have the same meaning, this is not captured here)
* Sequence information between words in a sentece is also lost, based on the context a word might have different meaning.

We'll have huge dimension vector, to reduce this we can remove the stopwords, stem the words.

### Preprocessing

The below steps can imporve BoW vectors.

* Removal of stopwords - small dimension and meaningful vectors.
* Stemming - tasty, tasteful, taste will become three different words in BoW vector. These can be converted to their common form (taste). This is called stemming. PorterStemmer and SnowballStemmer are some of them. ignores gramatical meaning of words. (Beatiful, Beauty is stemmed to Beaut)
* Lemmatization - 1. Break similar sentences to it's root word, Break Sentences into words((group New York into a single word)).
* lower case(To make Pasta, pasta the same in vector dimension)

### Bi-grams, n-grams

Creation of d-dimension vector with unique words is unigram(single word) BoW.

Let's take two sample sentences:

1. The device is good and affordabale
2. The device is not good and affordable

If we remove the stop words and create an uni-gram BoW what'll we get:

1. device good affordable
2. device good affordable

yep even not is a stop word in nltk library, now if we create bow vectors for both these sentence and calculate similariy. We'll get they are same but that's not the case right? How can we overcome this...

Instead of using single word we can use two words for creating a bi-gram BoW. Bi-Grams for 1st sentence will be `The device`, `device is`, `is good` and so on. If we create vectors now we'll have not good captured in second vector. This is not present in first vector increasing the distance.

With Bi-grams, n-grams, we can capture part of sequence information lost in uni-gram vectors. But the down side as n-grams(n=1,2,3) the dimensionality will increase because there are tons of word conmbinations and sentences.

## TF-IDF

We'll now cover the weight calculation for each word in TF-IDF vector. The vector here is also same as BoW, d-dimension of uniqe words in the corpus. This is also a derivation of BoW with the how the vector is filled up changed with weights, instead of occurunces of words.

To calculate weight for a single word in the vector, we need:
1. Term Frequency(TF)
2. Inverse Term Frequency(IDF)

### Term Frequency

TF(word, document) = Number of times Word occurs in document/Total number of words in the document

* word occurence in a document can never be greater than total number of words, hence the result of division always lies between 0 and 1.
* This can also be considered a probabality and can be defined as 0 <= TF(word, document) <= 1.
* 1 will occur when only a single word is present in a document or all the words in the document are same.

### inverse Document Frequency

IDF(word, corpus) = log(Total Number of docs/Number of documents where the word occurs)
$\text{IDF}(t) = \log \left( \frac{N}{n_i} \right)$


* Here we use the entire corpus instead of single document compared to IDF
* Number of documents where the word occurs can never be greater than Total Number of docs
* If there are 100 docs, if the word is present in all 100 docs, ex the might be present in all docs.
IDF(the, corpus) = log(100/100) = log(1) = 0
* Log never has values below zero(>=0) and the $\left( \frac{N}{n_i} \right)$ is between 1 to N.
* As ${n_i}$ increases IDF decreases, what does this mean. IDF will be really low for very common words like stopwords and very high for rare words. ${IDF} \propto \frac{1}{n_i}$
* IDF >= 0.

### Weights

* The weight for a word is product of TF and IDF, that's hwy the name TF-IDF
* More importance to rarer words in corpus(IDF)
* More importance if a word is frequent in a Document(TF)

*But still we haven't achieved our semantic similarity objective with this method as well*.

### Why use log for IDF?

* Stasically, frequency of words in y axis vs words in x axis will be a power law or paretto distribution, we can convert this to gaussian distribution using box-cox transforms(logarithm)
* Intutivley, Assume the below two cases:
    * IDF for The which will be in all documents wil be 1
    * IDF for rare words like civilization, let's say for a 1000 document corpus, it's in 5 documents. 1000 / 5 = 250. Let's say TF is 0.7 = 175. log(250) * 0.7 -> 1.65.
    * We can see points here. Without log the domination of IDF over TF and scale difference with and without log. For distance calculation algorithms(our objective of semantic similarity is a distance calculation algorithm) scale will have a big impact. As it's simply put dot product or cosine similarity. Even assuming during backpropogation with these large numbers we might end up with exploding gradients. These aRe some intuitve reasons to use log.

## Word2Vec

Word2Vec models are trained through deep learning supervision task from raw text or matrix factorizaiton. It's benefits are:

* Dense embeddings(less zero's)
* Higher dimension(more quality data)
* Relationship are identified(man-->woman, king-->queen distance vectors will be neraly identical)

This converts only word to vec. What we need is to convert reviews(sentences) to vector.

1. Average W2V: Sum of all word vectors / Number of words
2. TfIdf W2V: Summation of t_word_i * word2vec(word_i) / Summation of t_word_i - We calculate tf-idf of all words in the sentence(this is t). 

## Now let's implement the code for all concepts above

### Bag of Words


```python
count_vect = CountVectorizer() # from scikit-learn
final_counts = count_vect.fit_transform(final["Text"].values)
```


```python
type(final_counts)
```




    scipy.sparse._csr.csr_matrix




```python
final_counts.get_shape()
```




    (364171, 115281)




```python
final_counts[0].shape
```




    (1, 115281)




```python
print(final_counts[0])
```

    <Compressed Sparse Row sparse matrix of dtype 'int64'
    	with 53 stored elements and shape (1, 115281)>
      Coords	Values
      (0, 103749)	3
      (0, 113004)	1
      (0, 64507)	1
      (0, 22082)	3
      (0, 66253)	1
      (0, 71724)	2
      (0, 96473)	2
      (0, 63059)	1
      (0, 10401)	1
      (0, 65167)	1
      (0, 86314)	2
      (0, 59284)	2
      (0, 57052)	2
      (0, 103373)	4
      (0, 25403)	1
      (0, 9973)	1
      (0, 111527)	1
      (0, 85813)	1
      (0, 39477)	1
      (0, 7529)	1
      (0, 8302)	2
      (0, 53557)	3
      (0, 7734)	1
      (0, 24971)	1
      (0, 94619)	1
      :	:
      (0, 111991)	1
      (0, 57417)	1
      (0, 39520)	1
      (0, 89722)	1
      (0, 65217)	1
      (0, 7296)	2
      (0, 72824)	1
      (0, 113360)	1
      (0, 58762)	1
      (0, 94431)	1
      (0, 74846)	1
      (0, 59142)	2
      (0, 28971)	1
      (0, 7750)	1
      (0, 112660)	1
      (0, 104542)	2
      (0, 20386)	1
      (0, 112630)	1
      (0, 98814)	1
      (0, 19419)	1
      (0, 5093)	1
      (0, 47909)	1
      (0, 68341)	1
      (0, 112121)	1
      (0, 29981)	1


*Inferences from code*

* We can see 364K reviews with dimension of 115281, 115281 is the number of unique words in 364k+ reviews.
* Space complexity for storing a matrix is $O(m*n)$ where m=364171(y axis) and n=1152819(x axis). For a single review 90% or more than that will all be zeros.
* This complexity is really large just for zeros which is a huge waste of memory. 
* To overcome this, CountVectorizer implementation does this:--> Stores a dict of index(row no, col no) and count of k non zero values. Type - `csr_matrix`
* Now the complexiety reduces to $O(row no, col no, count)$. $O(m*n)$ -> $O(3*k)$ --> $O(k)$
* This implementation efficiency is directly proportational to sparsity of the matrix
* Sparsity is more number of zeros in the vector. To calculate sparsity -> number of non zero cells / m * n


```python
# Sparsity of a single element
# There are 53 non zero values in final_counts[0]
sparsity = 53 / (final_counts.shape[0] * final_counts.shape[1])
print(sparsity)
```

    1.262445898788837e-09


### PreProcessing Steps

We'll implement a list of steps:
1. Check if num is alphanum
2. Check word length > 2
3. stem word
4. Remove html tags
5. Remove punctuations
6. Check if word is stop word or not


```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop = set(stopwords.words("english")) # stopwords
sno = nltk.stem.SnowballStemmer("english") # Stemmer

def clean_html(sentence):
    """_summary_
    Clean html tags from a sentence
    Args:
        sentence (str): Passage of NLT
    """
    pattern = re.compile("<.*?>") # Capture html pattern
    cleaned_text = re.sub(pattern, "  ", sentence) # replace this with double space instead of space, because we'll split the sentence by space to convert it to words
    return cleaned_text

def clean_punctuation(sentence):
    """_summary_
    Clean ?,!,',",#,.,comma,),(,\,/
    Args:
        sentence (str): Passage of NLT
    """
    pattern_1 = r'[?|!|\'|"|#|]'
    pattern_2 = r'[.|,|)|(|\|/]'
    cleaned_text = re.sub(pattern_1, r"", sentence)
    cleaned_text = re.sub(pattern_2, " ", cleaned_text)
    return cleaned_text
```

    <>:20: SyntaxWarning: invalid escape sequence '\,'
    <>:20: SyntaxWarning: invalid escape sequence '\,'
    /var/folders/s2/zc28s499001f26bz7nbfmfhr0000gn/T/ipykernel_907/429179674.py:20: SyntaxWarning: invalid escape sequence '\,'
      """_summary_



```python
# Let's clean the text
final_sentences = []
all_positive_words = []
all_negative_words = []

# Iterate through text
for idx, category, text in final[["Score", "Text"]].itertuples():
    # Store words for a single review
    final_sentence = []
    # Clean html
    html_cleaned = clean_html(sentence=text)
    for word in html_cleaned.split():
        for cleaned_word in clean_punctuation(word).split():
            # Check alphanum, word > 2 and word not in stop word
            if (cleaned_word.isalpha()) and (len(cleaned_word) > 2) and (cleaned_word.lower() not in stop):
                # Stem word
                stemmed_word = sno.stem(cleaned_word.lower()).encode("utf8")
                final_sentence.append(stemmed_word)
                if category == "positive":
                    all_positive_words.append(stemmed_word)
                else:
                    all_negative_words.append(stemmed_word)
            else:
                continue
    final_sentences.append(b" ".join(final_sentence))
    
```


```python
final["cleanedText"] = final_sentences
```


```python
final.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>ProductId</th>
      <th>UserId</th>
      <th>ProfileName</th>
      <th>HelpfulnessNumerator</th>
      <th>HelpfulnessDenominator</th>
      <th>Score</th>
      <th>Time</th>
      <th>Summary</th>
      <th>Text</th>
      <th>cleanedText</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138706</th>
      <td>150524</td>
      <td>0006641040</td>
      <td>ACITT7DI6IDDL</td>
      <td>shari zychinski</td>
      <td>0</td>
      <td>0</td>
      <td>positive</td>
      <td>939340800</td>
      <td>EVERY book is educational</td>
      <td>this witty little book makes my son laugh at l...</td>
      <td>b'witti littl book make son laugh loud recit c...</td>
    </tr>
    <tr>
      <th>138688</th>
      <td>150506</td>
      <td>0006641040</td>
      <td>A2IW4PEEKO2R0U</td>
      <td>Tracy</td>
      <td>1</td>
      <td>1</td>
      <td>positive</td>
      <td>1194739200</td>
      <td>Love the book, miss the hard cover version</td>
      <td>I grew up reading these Sendak books, and watc...</td>
      <td>b'grew read sendak book watch realli rosi movi...</td>
    </tr>
    <tr>
      <th>138689</th>
      <td>150507</td>
      <td>0006641040</td>
      <td>A1S4A3IQ2MU7V4</td>
      <td>sally sue "sally sue"</td>
      <td>1</td>
      <td>1</td>
      <td>positive</td>
      <td>1191456000</td>
      <td>chicken soup with rice months</td>
      <td>This is a fun way for children to learn their ...</td>
      <td>b'fun way children learn month year learn poem...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store final table into an SqlLite table for future
conn = sqlite3.connect("final.sqlite")
cursor = conn.cursor()
conn.text_factory = str
final.to_sql(
    "Reviews",
    conn,
    schema=None,
    if_exists="replace",
)
```




    364171



### Bi-Gram, N-Gram


```python

```


```python
all_positive_words_count = {}
for positive_word in all_positive_words:
    word = positive_word.decode()
    all_positive_words_count[word] = all_positive_words_count.get(word, 0) + 1
all_negative_words_count = {}
for negative_word in all_negative_words:
    # Decode to convert strings frmo binary string to text
    word = negative_word.decode()
    all_negative_words_count[word] = all_negative_words_count.get(word, 0) + 1

# Sorting the count values
all_positive_words_count = sorted(all_positive_words_count.items(), key=lambda item: item[1], reverse=True)
all_negative_words_count = sorted(all_negative_words_count.items(), key=lambda item: item[1], reverse=True)
```


```python
all_positive_words_count[:5], all_negative_words_count[:5]
```




    ([('like', 139429),
      ('tast', 129047),
      ('good', 112766),
      ('flavor', 109624),
      ('love', 107357)],
     [('tast', 34585),
      ('like', 32330),
      ('product', 28218),
      ('one', 20569),
      ('flavor', 19575)])



Look at the words common in negative and positive words(like, flavor). Intuitivley this is bad as there's no clear distinction between positive and negative reviews. By having bigrams or N-grams we can improve this.

### Bi-Gram or N-grams


```python
# We'll use the n_gram parameter in CountVectorizer
# n_min and n_max, if this is 1,4. The vectorizer will create unigram, bigram, trigram and tetragram. As n_max increases the dimension(vocab size) of vector will also increase.
# Let's see this in action
bigram_vectorizer = CountVectorizer(
    ngram_range=(1,2),
)
bigram_final = bigram_vectorizer.fit_transform(final["Text"].values)
```


```python
bigram_final.shape
```




    (364171, 2910192)




```python
# Let's see the increase in feature
print(f"Bigram counts dimension is {bigram_final.shape[1]/final_counts.shape[1]}x of BagOfWords dimension")
```

    Bigram counts dimension is 25.24433341140344x of BagOfWords dimension


Woah! We've a 25x increase in dimension for just unigram to bigram.

### TfIdf

To implement TfIdf, we can just use the TfIdfVectorizer form scikit-learn.



```python
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf_counts = tfidf_vectorizer.fit_transform(final["Text"].values)
```


```python
tfidf_counts.shape
```




    (364171, 2910192)




```python
bigram_final.shape[1] == tfidf_counts.shape[1]
```




    True




```python
# Get tfidf features
features = tfidf_vectorizer.get_feature_names_out()
len(features)
```




    2910192



Features has the same length as dimensions, Assume this as a row vector all possible unique unigram and bigrams.


```python
features[10000:10010]
```




    array(['14 count', '14 country', '14 crackers', '14 credit', '14 crude',
           '14 cup', '14 cupcakes', '14 cups', '14 currants', '14 dad'],
          dtype=object)




```python
# Let's write a function to see top 25 highest features/dimensions in a review
def tfidf_top(review_counts, features, top_k=25):
    """_summary_
    Function to get top k TfIdf Score features(unigram, bigram in our case)
    Args:
        review_counts (np.array): review arrary
        features (_type_): TfIdf Features
        top_k (int, optional): _description_. Defaults to 25.
    """

    # Get top score indexesx, reverse sort them, get top_k values
    top_indexes = np.argsort(review_counts)[::-1][:top_k]
    top_feats = [(features[i], review_counts[i]) for i in top_indexes]
    df = pd.DataFrame(top_feats)
    df.columns = ["features", "scores"]
    return df
```


```python
# we'll convert a single review csr_matrix to array for sorting to the function above
tfidf_counts[0, :], tfidf_counts[0, :].toarray(), tfidf_counts[0, :].toarray().shape, tfidf_counts[0, :].toarray()[0], tfidf_counts[0, :].toarray()[0].shape
```




    (<Compressed Sparse Row sparse matrix of dtype 'float64'
     	with 122 stored elements and shape (1, 2910192)>,
     array([[0., 0., 0., ..., 0., 0., 0.]]),
     (1, 2910192),
     array([0., 0., 0., ..., 0., 0., 0.]),
     (2910192,))




```python
df = tfidf_top(review_counts=tfidf_counts[0, :].toarray()[0], features=features)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>recite</td>
      <td>0.234097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>book</td>
      <td>0.193848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>roses love</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>3</th>
      <td>about whales</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>introduces and</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>5</th>
      <td>recite from</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>6</th>
      <td>classic book</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>7</th>
      <td>book introduces</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>8</th>
      <td>whales india</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>9</th>
      <td>son laugh</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>10</th>
      <td>recite it</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>11</th>
      <td>silliness of</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>12</th>
      <td>at loud</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>13</th>
      <td>the silliness</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>14</th>
      <td>refrain he</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>15</th>
      <td>india drooping</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>16</th>
      <td>this witty</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>17</th>
      <td>memory when</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>18</th>
      <td>loud recite</td>
      <td>0.129413</td>
    </tr>
    <tr>
      <th>19</th>
      <td>new words</td>
      <td>0.125411</td>
    </tr>
    <tr>
      <th>20</th>
      <td>the refrain</td>
      <td>0.125411</td>
    </tr>
    <tr>
      <th>21</th>
      <td>witty little</td>
      <td>0.125411</td>
    </tr>
    <tr>
      <th>22</th>
      <td>to recite</td>
      <td>0.125411</td>
    </tr>
    <tr>
      <th>23</th>
      <td>drooping roses</td>
      <td>0.125411</td>
    </tr>
    <tr>
      <th>24</th>
      <td>book makes</td>
      <td>0.120369</td>
    </tr>
  </tbody>
</table>
</div>



This dataframe can be used to analyze classification results heuristically. Like we can find really bad samples of classifcation and investigate their scores.

### Word2Vec


```python
# We're gonna use google's new article word2vec model. This is a 3.3Gb file which will occupy 9Gb ram. This will not fit in my 8Gb machine. Let's download a picked version of this.
# From here - https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# We can load the model using genism
from gensim.models import Word2Vec, KeyedVectors
model = KeyedVectors.load_word2vec_format(
    "models/GoogleNews-vectors-negative300.bin", # Model path or name
    binary=True, # Since we're using .bin
)
```

This models is lookup table for vectors for their respective words.


```python
# Let's do some similarity checks for a word
model.most_similar("tasty")
```




    [('delicious', 0.8730390071868896),
     ('scrumptious', 0.8007041215896606),
     ('yummy', 0.7856924533843994),
     ('flavorful', 0.7420164346694946),
     ('delectable', 0.7385422587394714),
     ('juicy_flavorful', 0.7114803791046143),
     ('appetizing', 0.7017217874526978),
     ('crunchy_salty', 0.7012301087379456),
     ('flavourful', 0.691221296787262),
     ('flavoursome', 0.6857702732086182)]




```python
# Compare words
model.similarity("tasty", "delicious")
```




    0.873039




```python
# Let's try to find the vector for a word
sample_vector = model.get_vector("Tasty")
print(f"Shape of vector: {sample_vector.shape}")
print(f"Sample vector:\n {sample_vector}")
```

    Shape of vector: (300,)
    Sample vector:
     [-2.39257812e-01 -1.25000000e-01 -2.07031250e-01  2.16796875e-01
     -3.19824219e-02  5.12695312e-02  8.78906250e-02 -1.43554688e-01
     -1.38671875e-01 -3.41796875e-02 -5.56640625e-02  1.50390625e-01
      1.85546875e-01  1.17187500e-01 -2.85156250e-01  1.67968750e-01
     -1.01562500e-01 -1.01562500e-01  7.08007812e-02  6.68945312e-02
      2.28515625e-01 -8.78906250e-02  3.06640625e-01  1.34765625e-01
      2.42919922e-02 -1.54296875e-01 -2.92968750e-01  2.75390625e-01
      2.00195312e-01  1.42822266e-02 -2.69531250e-01 -2.63671875e-01
      3.65234375e-01 -1.55273438e-01 -3.02734375e-01 -3.22265625e-02
      2.77343750e-01 -9.22851562e-02 -1.07421875e-01  1.25976562e-01
      7.71484375e-02 -2.25585938e-01 -1.67968750e-01  3.33984375e-01
     -1.62109375e-01 -4.86328125e-01 -1.55273438e-01  1.07910156e-01
      7.32421875e-02  1.23535156e-01 -1.80664062e-01  2.85156250e-01
      1.92382812e-01  5.02929688e-02 -4.37011719e-02  1.90429688e-01
      1.70898438e-01 -9.57031250e-02 -1.53320312e-01 -6.98852539e-03
     -7.27539062e-02  8.25195312e-02 -1.64062500e-01  7.72094727e-03
     -8.20312500e-02 -4.29687500e-01  1.19628906e-01 -8.17871094e-03
     -1.53320312e-01  7.66601562e-02  3.69140625e-01 -1.74804688e-01
      1.26953125e-01  1.79687500e-01 -1.61132812e-01 -1.41601562e-01
     -3.00781250e-01 -3.07617188e-02 -9.22851562e-02  4.63867188e-02
     -4.73022461e-03  1.69677734e-02 -4.24804688e-02 -7.51953125e-02
     -7.37304688e-02 -1.55273438e-01 -5.23437500e-01  4.92187500e-01
      2.05078125e-02 -2.40234375e-01  5.66406250e-02 -7.27539062e-02
     -5.93261719e-02 -2.92968750e-01 -2.67578125e-01  8.25195312e-02
      1.05590820e-02  4.80957031e-02  9.96093750e-02  9.76562500e-03
     -1.17187500e-01 -9.32617188e-02 -4.46777344e-02  2.28515625e-01
     -9.47265625e-02 -2.83203125e-01 -2.12402344e-02 -2.14843750e-01
     -3.94531250e-01 -8.49609375e-02 -2.99072266e-02  1.66992188e-01
     -1.08886719e-01 -3.71093750e-01  3.39355469e-02 -6.22558594e-02
      4.78515625e-02 -8.30078125e-03 -2.91015625e-01  3.54003906e-03
      2.59765625e-01  8.25195312e-02  1.90429688e-01 -1.54296875e-01
     -2.39257812e-01  2.59765625e-01 -2.98828125e-01  1.82617188e-01
      7.37304688e-02 -4.95605469e-02 -2.09960938e-01  4.29687500e-02
      2.30468750e-01  2.06054688e-01 -3.55468750e-01  3.86718750e-01
     -1.21093750e-01 -1.68457031e-02  2.96875000e-01  3.24218750e-01
      7.95898438e-02 -1.87500000e-01  3.46679688e-02 -1.26953125e-01
     -1.11328125e-01 -2.71484375e-01  7.22656250e-02  8.93554688e-02
     -1.36718750e-01 -3.78417969e-02 -1.54296875e-01 -6.59179688e-02
      1.07421875e-01 -1.66015625e-01  5.71289062e-02 -4.95605469e-02
      1.48437500e-01 -9.47265625e-02 -2.55859375e-01 -3.54003906e-02
     -2.63671875e-01 -1.12792969e-01  3.20312500e-01  9.09423828e-03
      1.06933594e-01 -2.85156250e-01 -6.64062500e-02 -1.41601562e-01
     -7.86132812e-02 -1.24511719e-01 -1.29882812e-01 -9.03320312e-02
      1.84570312e-01 -2.20703125e-01 -4.41406250e-01  3.18359375e-01
     -1.69921875e-01  1.84570312e-01  1.96289062e-01 -2.20947266e-02
      2.89062500e-01 -2.71484375e-01 -4.27246094e-02 -1.28906250e-01
     -1.38671875e-01  8.78906250e-02 -4.12597656e-02  1.18164062e-01
     -4.88281250e-02 -1.31835938e-01 -1.41601562e-01  7.37304688e-02
      2.04101562e-01 -2.87109375e-01 -1.13281250e-01  3.20312500e-01
     -7.22656250e-02 -2.35351562e-01  5.93261719e-02 -6.39648438e-02
     -1.01562500e-01 -4.66308594e-02 -7.12890625e-02  2.14843750e-01
     -4.24804688e-02 -2.63671875e-01 -7.12890625e-02  9.91210938e-02
     -1.52587891e-02  1.89453125e-01 -2.21679688e-01  9.91210938e-02
     -4.05883789e-03  3.86718750e-01  1.53320312e-01 -1.41601562e-01
      1.55273438e-01  7.61718750e-02  1.33789062e-01 -1.70898438e-01
      5.63964844e-02  2.57812500e-01 -3.00781250e-01  4.51660156e-02
     -3.11279297e-02  1.07910156e-01 -3.20312500e-01 -6.25000000e-02
      1.08032227e-02  1.14746094e-01 -1.25976562e-01  1.75781250e-01
      4.71191406e-02  3.26171875e-01  3.90625000e-01 -4.06250000e-01
      2.69531250e-01  1.90429688e-01 -3.96484375e-01  9.47265625e-02
      4.62890625e-01 -1.88476562e-01 -6.03027344e-02 -8.93554688e-02
      2.45117188e-01 -6.88476562e-02 -3.56445312e-02  2.25585938e-01
      1.23901367e-02  1.34765625e-01 -2.65625000e-01  8.78906250e-02
      1.88476562e-01  1.56250000e-01  4.34570312e-02 -7.76367188e-02
     -2.03125000e-01 -8.23974609e-03 -2.73437500e-02 -1.22558594e-01
      7.22656250e-02  8.23974609e-03 -2.92968750e-01  7.47070312e-02
      2.53677368e-04  5.68847656e-02 -3.06640625e-01 -8.98437500e-02
     -2.38281250e-01  3.46679688e-02  5.46875000e-01  2.31445312e-01
     -2.81250000e-01  1.00097656e-01  3.86718750e-01  1.46484375e-01
     -1.29882812e-01 -3.16406250e-01 -1.77001953e-02 -1.74804688e-01
     -1.16210938e-01  1.44531250e-01 -2.48046875e-01  1.45507812e-01
     -2.22656250e-01  1.18652344e-01 -1.80664062e-01  1.00585938e-01
     -1.52343750e-01  3.90625000e-01  1.45507812e-01 -5.10253906e-02
     -1.76757812e-01  9.81445312e-02  2.50000000e-01 -1.41601562e-01
     -1.26342773e-02 -9.57031250e-02  6.29882812e-02 -1.55273438e-01]



```python
# Few stemmed words might not be present int the model
model.get_vector("tasti")
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[135], line 2
          1 # Few stemmed words might not be present int the model
    ----> 2 model.get_vector("tasti")


    File ~/miniconda3/lib/python3.12/site-packages/gensim/models/keyedvectors.py:446, in KeyedVectors.get_vector(self, key, norm)
        422 def get_vector(self, key, norm=False):
        423     """Get the key's vector, as a 1D numpy array.
        424 
        425     Parameters
       (...)
        444 
        445     """
    --> 446     index = self.get_index(key)
        447     if norm:
        448         self.fill_norms()


    File ~/miniconda3/lib/python3.12/site-packages/gensim/models/keyedvectors.py:420, in KeyedVectors.get_index(self, key, default)
        418     return default
        419 else:
    --> 420     raise KeyError(f"Key '{key}' not present")


    KeyError: "Key 'tasti' not present"



```python
# To overcome this we can train our own word2vec model from our corpus
# To do this we need to pass data in this format [[word1,word2], [word1,word2]] -> [[sentence1], [sentence2]] -> Where each sentence is split into words
# We won't remove stopwords because, we'll lose not.
word2_vec_reviews = []
for review in final["Text"].values:
    filtered_sentence = []
    # Clean html
    html_cleaned_review = clean_html(review)
    # Split word
    for word in html_cleaned_review.split():
        # Split based on punc
        for cleaned_word in clean_punctuation(word).split():
            if (cleaned_word.isalpha()):
                filtered_sentence.append(cleaned_word)
            else:
                continue
    word2_vec_reviews.append(filtered_sentence)
```


```python
print(word2_vec_reviews[0])
print(" ".join(word2_vec_reviews[0]))
```

    ['this', 'witty', 'little', 'book', 'makes', 'my', 'son', 'laugh', 'at', 'loud', 'i', 'recite', 'it', 'in', 'the', 'car', 'as', 'were', 'driving', 'along', 'and', 'he', 'always', 'can', 'sing', 'the', 'refrain', 'hes', 'learned', 'about', 'whales', 'India', 'drooping', 'i', 'love', 'all', 'the', 'new', 'words', 'this', 'book', 'introduces', 'and', 'the', 'silliness', 'of', 'it', 'all', 'this', 'is', 'a', 'classic', 'book', 'i', 'am', 'willing', 'to', 'bet', 'my', 'son', 'will', 'STILL', 'be', 'able', 'to', 'recite', 'from', 'memory', 'when', 'he', 'is', 'in', 'college']
    this witty little book makes my son laugh at loud i recite it in the car as were driving along and he always can sing the refrain hes learned about whales India drooping i love all the new words this book introduces and the silliness of it all this is a classic book i am willing to bet my son will STILL be able to recite from memory when he is in college



```python
import gensim
w2v_model = gensim.models.Word2Vec(
    word2_vec_reviews, # List of list sentences
    min_count=5, # word has to occur 5 times to create a vector
    vector_size=50, # Dimension of the vector
    workers=2, # Number of workers to build this model
)
```


```python
# Build the vocabulary
w2v_model.build_vocab(word2_vec_reviews, progress_per=10000)
```


```python
# Train the model
w2v_model.train(word2_vec_reviews, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
```




    (636393154, 836628210)




```python
# Make the mode more memeory efficient
w2v_model.init_sims(replace=True)
```

    /var/folders/s2/zc28s499001f26bz7nbfmfhr0000gn/T/ipykernel_907/2419709253.py:2: DeprecationWarning: Call to deprecated `init_sims` (Gensim 4.0.0 implemented internal optimizations that make calls to init_sims() unnecessary. init_sims() is now obsoleted and will be completely removed in future versions. See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4).
      w2v_model.init_sims(replace=True)



```python
w2v_model.wv.get_vector("taste"), w2v_model.wv.get_vector("taste").shape
```




    (array([-0.16849469,  0.16382068, -0.00642152, -0.35406202, -0.0265453 ,
             0.32891715, -0.10842107, -0.00634135,  0.03764758, -0.19081114,
            -0.02937614, -0.08751374,  0.16647403,  0.079576  ,  0.2227623 ,
            -0.01321285, -0.26592404,  0.23118702,  0.0995752 ,  0.01758653,
            -0.09244239, -0.02822721,  0.15739277, -0.12555908, -0.08097733,
             0.14662708, -0.27878094,  0.14695078, -0.11488359,  0.00485369,
            -0.05462616, -0.20081028, -0.1395006 , -0.02717623,  0.10202011,
            -0.08236805, -0.009123  , -0.03384229,  0.1199767 , -0.12020289,
             0.21988836, -0.07583645,  0.08863829,  0.2128078 , -0.00474559,
            -0.00857432, -0.0487239 , -0.06127046, -0.1365328 ,  0.02739566],
           dtype=float32),
     (50,))



Now we've created a word2vec model for our corpus.

### Average Word2Vec


```python
# Average word to vec for a sentence -> Average of all word vectors in a sentence
avg_sent_vectors = []
# Using cleaned text
for sentence in final["Text"].values:
    # Initial sent_vector with zeros
    sent_vector = np.zeros(50)
    cnt_vectors = 0
    for word in sentence.split():
        # w2v_model.wv.index_to_key -> vocab
        if word in w2v_model.wv.index_to_key:
            word_vec = w2v_model.wv.get_vector(word)
            sent_vector += word_vec
            cnt_vectors += 1
    sent_vector /= cnt_vectors
    avg_sent_vectors.append(sent_vector)
```

### TfIdf Word2Vec

We can get IDF scores from tfidf model, calculate tf scores from the sentence for each word.


```python
# Getting idf scores for words
idf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
print(idf_dict)
```


```python
tfidf_sent_vectors = []
for sentence in final["Text"].values:
    # Initial sent_vector with zeros
    sent_vector = np.zeros(50)
    weight_sum = 0
    tf_sum = 0
    for word in sentence:
        if word in w2v_model.wv.index_to_key and word in idf_dict:
            # Calculate word2vec
            word_vec = w2v_model.wv.get_vector(word)
            # Calculate tfidf score
            tf_idf = idf_dict.get(word) * sentence.count(word) / len(sentence)
            # Summation of tfidf scores for average
            weight_sum += tf_idf
            # Summation of sent vector
            sent_vector += (tf_idf * word_vec)
    if weight_sum != 0:
        sent_vector /= weight_sum
    tfidf_sent_vectors.append(sent_vector)
```


```python

```
