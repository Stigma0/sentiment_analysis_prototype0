import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import FeatureExtractors
import classification

"""Process sentiment data by separating based on emotion classes, balancing if desired, and optionally removing neutral sentiment."""
def data_balancer(data, balanced=True, remove_neutral=True):
    # Separate the data based on emotion classes
    negative_data = data[data['airline_sentiment'] == 'negative']
    positive_data = data[data['airline_sentiment'] == 'positive']
    neutral_data = data[data['airline_sentiment'] == 'neutral']

    # Determine the desired number of instances per class (same as the positive class)
    desired_instances = len(positive_data)

    if balanced:
        # Undersample the negative class to match the desired number of instances
        undersampled_negative_data = resample(negative_data, replace=False, n_samples=desired_instances,
                                              random_state=42)
        # Undersample the neutral class to match the desired number of instances
        undersampled_neutral_data = resample(neutral_data, replace=False, n_samples=desired_instances,
                                             random_state=42)
        # Concatenate the undersampled negative, positive, and neutral classes
        processed_data = pd.concat([undersampled_negative_data, positive_data, undersampled_neutral_data])
    else:
        # Concatenate the original data without undersampling
        processed_data = pd.concat([negative_data, positive_data, neutral_data])

    if remove_neutral:
        # Remove neutral sentiment from processed data
        processed_data = processed_data[processed_data['airline_sentiment'] != 'neutral']

    return processed_data

def feature_extractor(processed_data):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(processed_data['clean_tweet'],
                                                                        processed_data['airline_sentiment'],
                                                                        test_size=0.2, random_state=42)
    # Perform feature extraction (e.g., TF-IDF vectorization)
    train_vectors, test_vectors = FeatureExtractors.tfidfvectorize(train_data, test_data)
    return train_vectors, test_vectors, train_labels, test_labels

    # # Perform classification (e.g., SVM)
    # classification.svm_classification(train_vectors, test_vectors, train_labels, test_labels)


def sentiment_classification_example():
    # Example dataset
    data = pd.DataFrame({
        'clean_tweet': [
            "@USAirways isn't with chocoLate Flight please... i m mmmelting http://t.co/jJdoSFYIBM",
            "@USAirways isn't with Vanilla icecream Plane please... it's hott http://t.co/jJdoSFYIBM"
        ],
        'airline_sentiment': ['negative', 'negative']
    })

    # Example labels
    train_labels = [1]
    test_labels = [1]

    # Example sentences to classify
    sentence1 = "@USAirways isn't with chocoLate Flight please... i m mmmelting http://t.co/jJdoSFYIBM"
    sentence2 = "@USAirways isn't with Vanilla icecream Plane please... it's hott http://t.co/jJdoSFYIBM"

    # Preprocess sentences
    processed_sentence1 = [FeatureExtractors.preprocess_sentence(sentence1)]
    processed_sentence2 = [FeatureExtractors.preprocess_sentence(sentence2)]

    # TF-IDF vectorization
    train_vectors, test_vectors = FeatureExtractors.tfidfvectorize(processed_sentence1, processed_sentence2)

    # Perform SVM classification
    classification.svm_classification(train_vectors, test_vectors, train_labels, test_labels)


import optuna


def perform_hyperparameter_optimization(processed_data, n_trials=100):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(processed_data['clean_tweet'],
                                                                        processed_data['airline_sentiment'],
                                                                        test_size=0.2, random_state=42)
    # Perform feature extraction (e.g., TF-IDF vectorization)
    train_vectors, test_vectors = FeatureExtractors.tfidfvectorize(train_data, test_data)

    study = optuna.create_study(direction='maximize')
    study.set_user_attr('verbose', True)

    def objective(trial):
        return classification.objective(trial, train_vectors, test_vectors, train_labels, test_labels)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    print('Best hyperparameters:', best_params)
    print('Best objective value:', best_value)
