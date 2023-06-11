from collections import Counter

import nltk
import pandas as pd
from nltk import word_tokenize, ngrams
from wordcloud import WordCloud

tableau10 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213)]
for i in range(len(tableau10)):
    r, g, b = tableau10[i]
    tableau10[i] = (r / 255., g / 255., b / 255.)


def frequency_distribution(data, target_column):
    """
    Generates and displays frequency distributions of positive and negative words.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the tweet text.

    Returns:
        None
    """
    # Separate positive and negative data
    positive_tweets = data[data['airline_sentiment'] == 'positive']
    negative_tweets = data[data['airline_sentiment'] == 'negative']

    # Tokenize the text for positive and negative tweets
    positive_words = nltk.word_tokenize(' '.join(positive_tweets[target_column]))
    negative_words = nltk.word_tokenize(' '.join(negative_tweets[target_column]))

    # Count the frequency of positive and negative words
    positive_word_freq = nltk.FreqDist(positive_words)
    negative_word_freq = nltk.FreqDist(negative_words)

    # Get the most common words and their frequencies
    common_positive_words = positive_word_freq.most_common(30)
    common_negative_words = negative_word_freq.most_common(30)

    # Extract words and frequencies separately
    positive_words, positive_freq = zip(*common_positive_words)
    negative_words, negative_freq = zip(*common_negative_words)

    # Set up subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the frequency distribution for positive words
    ax1.bar(positive_words, positive_freq, color='green')
    ax1.set_title('Frequency Distribution of Positive Words')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    ax1.set_xticklabels(positive_words, rotation=45, ha='right')

    # Plot the frequency distribution for negative words
    ax2.bar(negative_words, negative_freq, color='red')
    ax2.set_title('Frequency Distribution of Negative Words')
    ax2.set_xlabel('Words')
    ax2.set_ylabel('Frequency')

    # Rotate x-axis labels for better readability
    ax2.set_xticklabels(negative_words, rotation=45, ha='right')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def histogram(data, target_column):
    """
    Generates and displays histograms of tweet lengths by sentiment class.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the tweet text.

    Returns:
        None
    """
    # Calculate the length of each tweet and assign it to a new column
    data['Tweet Length'] = data[target_column].apply(lambda x: len(nltk.word_tokenize(x)))

    # Separate the dataframe by sentiment class
    positive_tweets = data[data['airline_sentiment'] == 'positive']
    negative_tweets = data[data['airline_sentiment'] == 'negative']

    # Plot histogram for positive tweets
    plt.hist(positive_tweets['Tweet Length'], bins=10, alpha=0.5, label='Positive', color='green')

    # Plot histogram for negative tweets
    plt.hist(negative_tweets['Tweet Length'], bins=10, alpha=0.5, label='Negative', color='red')

    # Set labels and title for the plot
    plt.xlabel('Tweet Length')
    plt.ylabel('Count')
    plt.title('Histogram of Tweet Lengths by Sentiment')

    # Display a legend
    plt.legend()

    # Show the plot
    plt.show()


def compare_wordcloud(data, target_column):
    """
    Generates and displays word clouds for positive and negative sentiments.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the tweet text.

    Returns:
        None
    """
    # Concatenate the tweets for each sentiment category
    positive_tweets = ' '.join(data[data['airline_sentiment'] == 'positive'][target_column])
    negative_tweets = ' '.join(data[data['airline_sentiment'] == 'negative'][target_column])

    # Create word cloud for positive sentiment
    wordcloud_positive = WordCloud().generate(positive_tweets)

    # Create word cloud for negative sentiment
    wordcloud_negative = WordCloud().generate(negative_tweets)

    # Plot the word clouds
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title('Word Cloud - Positive Sentiment')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title('Word Cloud - Negative Sentiment')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def ngram_count_graph(data, target_column):
    # Combine positive and negative tweets for N-grams analysis
    positive_tweets = data[data['airline_sentiment'] == 'positive'][target_column]
    negative_tweets = data[data['airline_sentiment'] == 'negative'][target_column]

    # Tokenize the positive and negative tweets
    positive_tokens = [word_tokenize(tweet) for tweet in positive_tweets]
    negative_tokens = [word_tokenize(tweet) for tweet in negative_tweets]

    # Define the N-gram size
    n = 5  # Change this value to the desired N-gram size

    # Generate N-grams for positive tweets
    positive_n_grams = [ngrams(tokens, n) for tokens in positive_tokens]
    positive_n_grams = [gram for sublist in positive_n_grams for gram in sublist]

    # Generate N-grams for negative tweets
    negative_n_grams = [ngrams(tokens, n) for tokens in negative_tokens]
    negative_n_grams = [gram for sublist in negative_n_grams for gram in sublist]

    # Count the frequency of each N-gram for positive tweets
    positive_n_gram_freq = pd.Series(positive_n_grams).value_counts().reset_index()
    positive_n_gram_freq.columns = ['N-gram', 'Frequency']

    # Filter N-grams that occur more than once for positive tweets
    positive_n_gram_freq = positive_n_gram_freq[positive_n_gram_freq['Frequency'] > 1]

    # Count the frequency of each N-gram for negative tweets
    negative_n_gram_freq = pd.Series(negative_n_grams).value_counts().reset_index()
    negative_n_gram_freq.columns = ['N-gram', 'Frequency']

    # Filter N-grams that occur more than once for negative tweets
    negative_n_gram_freq = negative_n_gram_freq[negative_n_gram_freq['Frequency'] > 1]

    # Plot the top N-grams for positive tweets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(positive_n_gram_freq['N-gram'].astype(str)[:10], positive_n_gram_freq['Frequency'][:10])
    plt.title('Top N-grams - Positive Sentiment')
    plt.xlabel('N-gram')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    # Plot the top N-grams for negative tweets
    plt.subplot(1, 2, 2)
    plt.bar(negative_n_gram_freq['N-gram'].astype(str)[:10], negative_n_gram_freq['Frequency'][:10])
    plt.title('Top N-grams - Negative Sentiment')
    plt.xlabel('N-gram')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def ngram_count_unique_graph(data, target_column):
    """
    Generates and displays a bar graph showing the counts of unique N-grams.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the text data.

    Returns:
        None
    """
    # Tokenize the text column
    data['tokens'] = data[target_column].apply(lambda x: x.split())

    # Define the N-gram sizes
    n_values = [1, 2, 3, 4]  # unigrams, bigrams, trigrams, quadgrams

    # Count the frequency of each N-gram
    n_gram_counts = {}

    # Iterate over each N-gram size
    for n in n_values:
        # Generate N-grams for each row in the dataframe
        n_grams = data['tokens'].apply(lambda x: list(ngrams(x, n)))

        # Flatten the list of N-grams
        flattened_n_grams = [gram for sublist in n_grams for gram in sublist]

        # Count the frequency of each N-gram
        n_gram_freq = pd.Series(flattened_n_grams).value_counts().reset_index()
        n_gram_freq.columns = ['N-gram', 'Frequency']

        # Filter out unique N-grams (occurring only once)
        unique_n_grams = n_gram_freq[n_gram_freq['Frequency'] == 1]

        # Store the counts of unique N-grams
        n_gram_counts[n] = unique_n_grams.shape[0]

    # Plot the N-gram counts
    plt.bar(n_gram_counts.keys(), n_gram_counts.values())
    plt.xlabel('N-gram Size')
    plt.ylabel('Count')
    plt.title('Unique N-gram Counts')
    plt.show()


def sentiment_word_count(data, target_column='clean_tweet'):
    """
    Generates and displays bar graphs showing the word counts for negative and positive sentiments.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the text data (default: 'clean_tweet').

    Returns:
        None
    """
    # Separate the data into negative and positive dataframes
    negative_df = data[data['airline_sentiment'] == 'negative']
    positive_df = data[data['airline_sentiment'] == 'positive']

    # Preprocess the tweets and count the words for the negative dataframe
    negative_words = ' '.join(negative_df[target_column]).lower().split()
    negative_word_counts = Counter(negative_words)

    # Preprocess the tweets and count the words for the positive dataframe
    positive_words = ' '.join(positive_df[target_column]).lower().split()
    positive_word_counts = Counter(positive_words)

    # Get the top N words with the highest frequencies
    top_n = 10
    top_negative_words = negative_word_counts.most_common(top_n)
    top_positive_words = positive_word_counts.most_common(top_n)

    # Plot the word count graphs for negative and positive data
    plt.figure(figsize=(10, 6))

    # Negative word count graph
    plt.subplot(1, 2, 1)
    plt.bar([word for word, count in top_negative_words], [count for word, count in top_negative_words])
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Top {} Words in Negative Tweets'.format(top_n))
    plt.xticks(rotation=45)

    # Positive word count graph
    plt.subplot(1, 2, 2)
    plt.bar([word for word, count in top_positive_words], [count for word, count in top_positive_words])
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Top {} Words in Positive Tweets'.format(top_n))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def display_rare_words(data, num_words=100, target_column='clean_tweet'):
    """
    Displays rare words occurring only once in the text data.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        num_words (int): Number of rare words to display (default: 100).
        target_column (str): Name of the target column containing the text data (default: 'clean_tweet').

    Returns:
        None
    """
    # Preprocess the tweets and count the words
    words = ' '.join(data[target_column]).split()
    word_counts = Counter(words)

    # Get the rare occurring words (occurring only once)
    rare_words = [word for word, count in word_counts.items() if count == 1]

    # Display the rare words
    num_rare_words = len(rare_words)
    if num_rare_words > num_words:
        rare_words = rare_words[:num_words]

    for word in rare_words:
        print(word)


def ngram_count_unique_graph_pos_neg(data, target_column):
    """
    Generates and displays a bar graph showing the counts of unique N-grams for positive and negative sentiments.

    Args:
        data (pandas.DataFrame): DataFrame containing the data.
        target_column (str): Name of the target column containing the text data.

    Returns:
        None
    """
    # Tokenize the text column
    data['tokens'] = data[target_column].apply(lambda x: x.split())

    # Separate positive and negative data
    positive_data = data[data['airline_sentiment'] == 'positive']
    negative_data = data[data['airline_sentiment'] == 'negative']

    # Define the N-gram sizes
    n_values = [1, 2, 3, 4]  # unigrams, bigrams, trigrams, quadgrams

    # Count the frequency of each N-gram for positive and negative data
    positive_n_gram_counts = {}
    negative_n_gram_counts = {}

    # Iterate over each N-gram size
    for n in n_values:
        # Generate N-grams for positive data
        positive_n_grams = positive_data['tokens'].apply(lambda x: list(ngrams(x, n)))

        # Flatten the list of N-grams for positive data
        flattened_positive_n_grams = [gram for sublist in positive_n_grams for gram in sublist]

        # Count the frequency of each N-gram for positive data
        positive_n_gram_freq = pd.Series(flattened_positive_n_grams).value_counts().reset_index()
        positive_n_gram_freq.columns = ['N-gram', 'Frequency']

        # Filter out unique N-grams (occurring only once) for positive data
        unique_positive_n_grams = positive_n_gram_freq[positive_n_gram_freq['Frequency'] == 1]

        # Store the counts of unique N-grams for positive data
        positive_n_gram_counts[n] = unique_positive_n_grams.shape[0]

        # Generate N-grams for negative data
        negative_n_grams = negative_data['tokens'].apply(lambda x: list(ngrams(x, n)))

        # Flatten the list of N-grams for negative data
        flattened_negative_n_grams = [gram for sublist in negative_n_grams for gram in sublist]

        # Count the frequency of each N-gram for negative data
        negative_n_gram_freq = pd.Series(flattened_negative_n_grams).value_counts().reset_index()
        negative_n_gram_freq.columns = ['N-gram', 'Frequency']

        # Filter out unique N-grams (occurring only once) for negative data
        unique_negative_n_grams = negative_n_gram_freq[negative_n_gram_freq['Frequency'] == 1]

        # Store the counts of unique N-grams for negative data
        negative_n_gram_counts[n] = unique_negative_n_grams.shape[0]

    # Plot the N-gram counts for positive and negative data side by side
    fig, ax = plt.subplots(figsize=(12, 6))

    # Positive N-gram counts
    ax.bar(positive_n_gram_counts.keys(), positive_n_gram_counts.values(), width=0.4, align='edge', alpha=0.5,
           label='Positive', color='green')

    # Negative N-gram counts
    ax.bar(negative_n_gram_counts.keys(), negative_n_gram_counts.values(), width=-0.4, align='edge', alpha=0.5,
           label='Negative', color='red')

    # Set labels and title for the plot
    ax.set_xlabel('N-gram Size')
    ax.set_ylabel('Count')
    ax.set_title('Unique N-gram Counts by Sentiment')

    # Display a legend
    ax.legend()

    # Show the plot
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def display_metrics(results):
    """
    Displays a bar chart showing evaluation metrics for different datasets and sentiments.

    Args:
        results (dict): Dictionary containing the evaluation results.

    Returns:
        None
    """
    labels = ["Precision", "Recall", "F1-Score"]

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()

    # For positive sentiment
    rects1 = ax.bar(x - width / 2 - 0.2, [results["Dataset 1"][label]["Positive"] for label in labels], width,
                    label='Dataset 1 Positive', color=tableau10[0])
    rects2 = ax.bar(x - width / 2 - 0.1, [results["Dataset 2"][label]["Positive"] for label in labels], width,
                    label='Dataset 2 Positive', color=tableau10[1])

    # For negative sentiment
    rects3 = ax.bar(x + width / 2 + 0.1, [results["Dataset 1"][label]["Negative"] for label in labels], width,
                    label='Dataset 1 Negative', color=tableau10[2])
    rects4 = ax.bar(x + width / 2 + 0.2, [results["Dataset 2"][label]["Negative"] for label in labels], width,
                    label='Dataset 2 Negative', color=tableau10[3])

    # Add value labels on top of each bar
    def autolabel(rects, i):
        for rect in rects:
            height = rect.get_height()
            x_pos = rect.get_x() + rect.get_width() / 2.
            ax.text(x_pos + i, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    i = -0.02
    autolabel(rects1, i)
    i = -1 * i
    autolabel(rects2, i)
    i = -1 * i
    autolabel(rects3, i)
    i = -1 * i
    autolabel(rects4, i)

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Evaluation scores by dataset and sentiment')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.legend(loc=4)

    ax.set_ylim(0.5, 1)  # Set y-axis limits between 0.5 and 1
    ax.set_yticks(np.arange(0.5, 1.0, 0.1))  # Set y-axis tick positions

    fig.tight_layout()

    plt.show()
