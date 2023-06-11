import re

import contractions as contractions
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(data, mentions=0, stemming=1, new_column_name='clean_tweet'):
    """
    Preprocesses the text data in a DataFrame by removing URLs, mentions, non-alphabetic characters, duplicate letters,
    stopwords, and performing stemming.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        mentions (int): Flag indicating whether to remove mentions (0 - don't remove, 1 - remove) (default: 0).
        stemming (int): Flag indicating the stemming type to apply (0 - no stemming, 1 - Snowball stemmer,
                         2 - Porter stemmer) (default: 1).
        new_column_name (str): Name of the new column to store the preprocessed text (default: 'clean_tweet').

    Returns:
        pandas.DataFrame: DataFrame with the preprocessed text data and additional columns for mentions and hashtags.
    """
    def tweet_to_words(tweet):
        # Remove URLs
        pattern = r'\bhttp\w*\S+\b'
        tweet = re.sub(pattern, '', tweet)

        if mentions:
            # Remove mentions
            tweet = re.sub(r'@\w+', '', tweet)

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

        if stemming == 1:
            # Perform stemming
            stemmer = SnowballStemmer("english")
            stemmed_words = [stemmer.stem(word) for word in meaningful_words]
        elif stemming == 2:
            # Perform stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in meaningful_words]
        else:
            stemmed_words = meaningful_words

        # We can save preprocessing dataset
        return " ".join(stemmed_words)

    def extract_mentions(tweet):
        # Extract user mentions
        mentions = re.findall(r'@(\w+)', tweet)
        return mentions

    def extract_hashtags(tweet):
        hashtags = re.findall(r'#(\w+)', tweet)
        return hashtags

    # Extract mentions and create a new 'mentions' column
    data['mentions'] = data['text'].apply(extract_mentions)
    data['hashtags'] = data['text'].apply(extract_hashtags)
    data[new_column_name] = data['text'].apply(tweet_to_words)

    # Remove duplicate entries based on the 'text' column
    df_no_duplicates = data.drop_duplicates(subset=new_column_name)
    return df_no_duplicates


def tfidf_vectorize(train_data, test_data, ngram_range=(1, 3), max_features=None,
                    sublinear_tf=False, min_df=1, max_df=1.0):
    """
    Vectorizes the text data using TF-IDF representation.

    Args:
        train_data (list): List of training text data.
        test_data (list): List of test text data.
        ngram_range (tuple): Range of n-grams to consider (default: (1, 3)).
        max_features (int or None): Maximum number of features to keep (default: None).
        sublinear_tf (bool): Apply sublinear term frequency scaling (default: False).
        min_df (int or float): Minimum document frequency or proportion (default: 1).
        max_df (float): Maximum document frequency proportion (default: 1.0).

    Returns:
        tuple: A tuple containing the vectorized training data and test data.
    """
    # Create a TfidfVectorizer object
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                 sublinear_tf=sublinear_tf, min_df=min_df, max_df=max_df)

    # Preprocess the text data using the TfidfVectorizer
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    print("Important information about TfidfVectorizer:")
    print("Number of features:", train_vectors.shape[1])
    print("Max features:", vectorizer.max_features)
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    return train_vectors, test_vectors
