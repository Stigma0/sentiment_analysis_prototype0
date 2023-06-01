import nltk
from nltk import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def save_frequency_distribution_graph(data, column_name):
    # Tokenize the sentences into individual words
    data['tokenized_sentences'] = data[column_name].apply(nltk.word_tokenize)

    # Create a frequency distribution of the corpus
    freq_dist = FreqDist([word for sentence in data['tokenized_sentences'] for word in sentence])

    # Plot the frequency distribution
    plt.figure(figsize=(12, 8))
    freq_dist.plot(50, cumulative=False)
    plt.title('Frequency Distribution')
    plt.xlabel('Words')
    plt.ylabel('Frequency')


    # Save the graph
    plt.savefig(fname='FreqDist')
    plt.close()


def visualize_word_cloud(data, column_name):
    # Concatenate all sentences into a single string
    text = ' '.join(data[column_name])

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()