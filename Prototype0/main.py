import optuna
from sklearn.model_selection import train_test_split

import utility, classification, FeatureExtractors
import pandas as pd
import visualization
# Read the data from a CSV file
data = pd.read_csv('Tweets.csv')
data = FeatureExtractors.preprocessing(data)
data = utility.data_balancer(data)
train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data)
classification.svm_classification(train_vectors, test_vectors, train_labels, test_labels)
# study = optuna.create_study(direction='maximize')
# # Set verbosity for the trial
# study.set_user_attr('verbose', True)
# study.optimize(lambda trial: classification.objective(trial, train_vectors, test_vectors, train_labels, test_labels), n_trials=100)
# # Get the best hyperparameters and the best objective value
# best_params = study.best_params
# best_value = study.best_value
#
# print('Best hyperparameters:', best_params)
# print('Best objective value:', best_value)

# utility.process_sentiment_data(processed_data, True, True)
# processed_data = FeatureExtractors.preprocessing(data)
# visualization.save_frequency_distribution_graph(data,'clean_tweet')
# visualization.visualize_word_cloud(data,'clean_tweet')

