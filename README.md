# Combined_Intelligence

## NLP Submission - #buildwithai

This project seeks to understand popular sentiment concerning COVID-19 using data collected from 80,934 tweets.

See our presentation here: https://youtu.be/UaYQ-wA6WtY

Our viola dashboard is here: https://hub.gke.mybinder.org/user/jazmiahenry-com-ed_intelligence-iyd76ooy/voila/render/Code_Final/covid_sentiment.ipynb?token=Gn4lV0LHQ2a7Ida1jlOp4Q

*Code for presentation available in presentation folder*

In our project, we sought to answer a few simple questions:

    1. What are the most common words in the entire dataset?
    2. What are the most common words in the dataset for negative and positive tweets, 
    respectively?    
    3. Which trends are associated with my dataset?    
    4. Which trends are associated with either of the sentiments? 
    Are they compatible with the sentiments?
    
*Dataset credit: https//github.com/sandyskim/coronavirus-tweets*

## Strategies Used

### Data Cleaning

In any natural language processing task, cleaning raw text data is an important step. It helps in getting rid of the unwanted words and characters which helps in obtaining better features. If we skip this step then there is a higher chance that we are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry too much weight within the text.

We measured the length of tweets within our training and testing data. The visualization is followed:

![covid-sentiment_cell_17_output_0](https://user-images.githubusercontent.com/48301423/88496948-c1156f80-cf8c-11ea-8c73-7cef7580b18b.png)

### Filtering

We began by filtering out tweets that contain racist and sexist data. 

We found that 7% of the included tweets contained racist and sexist language and 93% of tweets in the dataset did not.

Next, we removed twitter handles (ensures privacy of users), punctuation, numbers, special characters, and short words to provide for more robust analysis.

Our last step in data filtering included normalizing our texts using tokenization.

### Findings

Using wordcloud visualization, we are able to map out most used words amongst Twitter users relating to COVID-19 as seen here:

![covid-sentiment_cell_39_output_0](https://user-images.githubusercontent.com/48301423/88497323-04241280-cf8e-11ea-9f1d-93e21f8e0ce8.png)

When we confine our analysis to only non-racist and sexist language, we find the most commonly used words to be these:

![covid-sentiment_cell_42_output_0](https://user-images.githubusercontent.com/48301423/88497385-36357480-cf8e-11ea-82fc-8b6721ac70dd.png)

As a comparison, tweets with racist and sexist language used the following words the most:

![covid-sentiment_cell_44_output_0](https://user-images.githubusercontent.com/48301423/88497445-5bc27e00-cf8e-11ea-83c3-874050824996.png)

### Impact of Hashtags on Tweet Sentiment

Hashtags in twitter are synonymous with the ongoing trends on twitter at any particular point in time. We should try to check whether these hashtags add any value to our sentiment analysis task, i.e., they help in distinguishing tweets into the different sentiments.

We store all the trend terms in two separate lists — one for non-racist/sexist tweets and the other for racist/sexist tweets.

Trend amongst non-racist or sexist tweets:

![covid-sentiment_cell_50_output_0](https://user-images.githubusercontent.com/48301423/88497542-ae039f00-cf8e-11ea-9a67-089622aba1d7.png)

Trend amonst racist or sexist tweets:

![covid-sentiment_cell_52_output_0](https://user-images.githubusercontent.com/48301423/88497584-cc699a80-cf8e-11ea-9a75-236dd9a223ef.png)

### Analyzing Preprocessed Data
To analyse a preprocessed data, it needs to be converted into features. Depending upon the usage, text features can be constructed using assorted techniques – Bag of Words, TF-IDF, and Word Embeddings.

#### Bag-of-words Feature

Let’s start with the Bag-of-Words Features.

Consider a Corpus C of D documents {d1,d2…..dD} and N unique tokens extracted out of the corpus C. The N tokens (words) will form a dictionary and the size of the bag-of-words matrix M will be given by D X N. Each row in the matrix M contains the frequency of tokens in document D(i).

Let us understand this using a simple example.

D1: He is a lazy boy. She is also lazy.

D2: Smith is a lazy person.

The dictionary created would be a list of unique tokens in the corpus =[‘He’,’She’,’lazy’,’boy’,’Smith’,’person’]

Here, D=2, N=6

The matrix M of size 2 X 6 will be represented as –

Now the columns in the above matrix can be used as features to build a classification model.

#### TF-IDF Features

This is another method which is based on the frequency method but it is different to the bag-of-words approach in the sense that it takes into account not just the occurrence of a word in a single document (or tweet) but in the entire corpus.

TF-IDF works by penalising the common words by assigning them lower weights while giving importance to words which are rare in the entire corpus but appear in good numbers in few documents.

Let’s have a look at the important terms related to TF-IDF:

TF = (Number of times term t appears in a document)/(Number of terms in the document)

IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.

TF-IDF = TF*IDF

#### Word2Vec Features

Word embeddings are the modern way of representing words as vectors. The objective of word embeddings is to redefine the high dimensional word features into low dimensional feature vectors by preserving the contextual similarity in the corpus. They are able to achieve tasks like King -man +woman = Queen, which is mind-blowing.

Drawing

The advantages of using word embeddings over BOW or TF-IDF are:

Dimensionality reduction - significant reduction in the no. of features required to build a model.

It capture meanings of the words, semantic relationships and the different types of contexts they are used in.


**Word2Vec Embeddings**

Word2Vec is not a single algorithm but a combination of two techniques – CBOW (Continuous bag of words) and Skip-gram model. Both of these are shallow neural networks which map word(s) to the target variable which is also a word(s). Both of these techniques learn weights which act as word vector representations.

CBOW tends to predict the probability of a word given a context. A context may be a single adjacent word or a group of surrounding words. The Skip-gram model works in the reverse manner, it tries to predict the context for a given word.

<img width="803" alt="word2vec" src="https://user-images.githubusercontent.com/48301423/88504998-3476ab80-cfa4-11ea-975a-ad5c2b3b090d.png">

There are three laters: - an input layer, - a hidden layer, and - an output layer.

The input layer and the output, both are one- hot encoded of size [1 X V], where V is the size of the vocabulary (no. of unique words in the corpus). The output layer is a softmax layer which is used to sum the probabilities obtained in the output layer to 1. The weights learned by the model are then used as the word-vectors.

We will go ahead with the Skip-gram model as it has the following advantages:

It can capture two semantics for a single word. i.e it will have two vector representations of ‘apple’. One for the company Apple and the other for the fruit.

Skip-gram with negative sub-sampling outperforms CBOW generally.

We will train a Word2Vec model on our data to obtain vector representations for all the unique words present in our corpus. There is one more option of using pre-trained word vectors instead of training our own model. Some of the freely available pre-trained vectors are:

Google News Word Vectors

Freebase names

DBPedia vectors (wiki2vec)

#### Doc2Vec Embedding

Doc2Vec model is an unsupervised algorithm to generate vectors for sentence/paragraphs/documents. This approach is an extension of the word2vec. The major difference between the two is that doc2vec provides an additional context which is unique for every document in the corpus. This additional context is nothing but another feature vector for the whole document. This document vector is trained along with the word vectors.

### Modeling

After completing all the pre-modeling stages required to get the data in the proper form and shape, build models on the datasets with different feature sets prepared in the earlier sections — Bag-of-Words, TF-IDF, word2vec vectors, and doc2vec vectors. We use the following algorithms to build models:

Logistic Regression
Support Vector Machine
RandomForest
XGBoost

Evaluation Metric

F1 score is being used as the evaluation metric. It is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is suitable for uneven class distribution problems.

The important components of F1 score are:

True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
False Positives (FP) – When actual class is no and predicted class is yes.
False Negatives (FN) – When actual class is yes but predicted class in no.

Precision = TP/TP+FP

Recall = TP/TP+FN

F1 Score = 2(Recall Precision) / (Recall + Precision)
