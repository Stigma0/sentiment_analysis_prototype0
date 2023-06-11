import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

import FeatureExtractors
import classification

"""Process sentiment data by separating based on emotion classes, balancing if desired, and optionally removing neutral sentiment."""


def data_balancer(data, balanced=True, process_neutral=True, remove_neutral=True):
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
        if process_neutral:
            # Undersample the neutral class to match the desired number of instances
            undersampled_neutral_data = resample(neutral_data, replace=False, n_samples=desired_instances,
                                                 random_state=42)
            # Concatenate the undersampled negative, positive, and neutral classes
            processed_data = pd.concat([undersampled_negative_data, positive_data, undersampled_neutral_data])
        else:
            processed_data = pd.concat([undersampled_negative_data, positive_data])
    else:
        # Concatenate the original data without undersampling
        processed_data = pd.concat([negative_data, positive_data, neutral_data])

    if remove_neutral:
        # Remove neutral sentiment from processed data
        processed_data = processed_data[processed_data['airline_sentiment'] != 'neutral']

    return processed_data


def feature_extractor(processed_data, target_column, ngram_range=(1, 3), max_features=None,
                      sublinear_tf=False, min_df=1, max_df=1.0):
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(processed_data[target_column],
                                                                        processed_data['airline_sentiment'],
                                                                        test_size=0.2, random_state=42)
    # Find and argument to make the spliting balanced
    # Perform feature extraction (e.g., TF-IDF vectorization)
    train_vectors, test_vectors = FeatureExtractors.tfidf_vectorize(train_data, test_data, ngram_range, max_features,
                                                                    sublinear_tf, min_df, max_df)
    return train_vectors, test_vectors, train_labels, test_labels

    # # Perform classification (e.g., SVM)
    # classification.svm_classification(train_vectors, test_vectors, train_labels, test_labels)


import optuna



def perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, n_trials=100):
    """
    Performs hyperparameter optimization using Optuna.

    Args:
        train_vectors (array-like): Training data vectors.
        test_vectors (array-like): Testing data vectors.
        train_labels (array-like): Training data labels.
        test_labels (array-like): Testing data labels.
        n_trials (int): Number of trials for optimization. Default is 100.

    Returns:
        tuple: Best hyperparameters (C, kernel, gamma).
    """
    study = optuna.create_study(direction='maximize')
    study.set_user_attr('verbose', True)

    def objective(trial):
        return classification.objective(trial, train_vectors, test_vectors, train_labels, test_labels)

    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    print('Best hyperparameters:', best_params)
    print('Best objective value:', best_value)

    return best_params['C'], best_params['kernel'], best_params['gamma']


def drop_columns(data):
    """
    Drops specified columns from the dataframe.

    Args:
        data (pandas.DataFrame): Input dataframe.

    Returns:
        pandas.DataFrame: Dataframe with specified columns dropped.
    """
    # List of columns to drop
    columns_to_drop = [
        'tweet_id',
        'airline_sentiment_confidence',
        'negativereason',
        'negativereason_confidence',
        'airline',
        'airline_sentiment_gold',
        'name',
        'negativereason_gold',
        'retweet_count',
        'tweet_coord',
        'tweet_created',
        'tweet_location',
        'user_timezone'
    ]

    # Drop the specified columns from the dataframe
    data = data.drop(columns_to_drop, axis=1)
    return data


def create_results():
    results = {
        "Dataset 1": {
            "Precision": {"Positive": 0.0, "Negative": 0.0},
            "Recall": {"Positive": 0.0, "Negative": 0.0},
            "F1-Score": {"Positive": 0.0, "Negative": 0.0},
        },
        "Dataset 2": {
            "Precision": {"Positive": 0.0, "Negative": 0.0},
            "Recall": {"Positive": 0.0, "Negative": 0.0},
            "F1-Score": {"Positive": 0.0, "Negative": 0.0},
        },
    }
    return results


def fill_results(dataset, report, results):
    """
    Updates the results dictionary with evaluation metrics for a given dataset.

    Args:
        dataset (str): Name of the dataset to be displayed.
        report (dict): Dictionary containing the evaluation report.
        results (dict): Dictionary containing the results to be updated.

    Returns:
        dict: Updated results dictionary.
    """
    results[dataset]["Precision"]["Positive"] = report["positive"]["precision"]
    results[dataset]["Precision"]["Negative"] = report["negative"]["precision"]
    results[dataset]["Recall"]["Positive"] = report["positive"]["recall"]
    results[dataset]["Recall"]["Negative"] = report["negative"]["recall"]
    results[dataset]["F1-Score"]["Positive"] = report["positive"]["f1-score"]
    results[dataset]["F1-Score"]["Negative"] = report["negative"]["f1-score"]

    return results

