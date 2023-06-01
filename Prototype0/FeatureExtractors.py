import re

import contractions as contractions
import nltk
import pandas as pd
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocessing(data):
    def tweet_to_words(tweet):
        # Remove URLs
        pattern = r'\bhttp\w*\S+\b'
        tweet = re.sub(pattern, '', tweet)

        # Convert to lowercase
        tweet = tweet.lower()

        # Expand contractions
        tweet = contractions.fix(tweet)

        # Remove non-alphabetic characters
        pattern = r"[^a-zA-Z]"
        tweet = re.sub(pattern, " ", tweet)

        # Remove duplicate letters
        pattern = r'(\w)\1+'
        tweet = re.sub(pattern, r'\1', tweet)

        # Tokenize the tweet
        tokens = nltk.word_tokenize(tweet)

        # Remove stopwords
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in tokens if not w in stops]

        # Perform stemming
        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(word) for word in meaningful_words]

        return " ".join(stemmed_words)

    data['clean_tweet'] = data['text'].apply(tweet_to_words)
    # Remove duplicate entries based on the 'text' column
    df_no_duplicates = data.drop_duplicates(subset='clean_tweet')
    return df_no_duplicates

def tfidfvectorize(train_data, test_data):
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(ngram_range=(1,9))
    # Preprocess the text data using the TfidfVectorizer
    train_vectors = vectorizer.fit_transform(train_data)
    # ngrams = vectorizer.vocabulary_
    # print(ngrams)
    test_vectors = vectorizer.transform(test_data)

    return train_vectors, test_vectors


def preprocess_sentence(sentence):
    def tweet_to_words(tweet):
        pattern = r'\bhttp\w*\S+\b'
        wout_links = re.sub(pattern, '', tweet)
        without_apostrophe_t = re.sub(r"\b(t)(\s+)", r" not ", wout_links)
        letters_only = re.sub("[^a-zA-Z]", " ", without_apostrophe_t)
        words = letters_only.lower()
        pattern = r'(\w)\1+'
        rm_dup_letters = re.sub(pattern, r'\1', words)
        corr_word = rm_dup_letters.split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in corr_word if not w in stops]

        stemmer = SnowballStemmer("english")
        stemmed_words = [stemmer.stem(word) for word in meaningful_words]

        return " ".join(stemmed_words)

    preprocessed_sentence = tweet_to_words(sentence)

    return preprocessed_sentence