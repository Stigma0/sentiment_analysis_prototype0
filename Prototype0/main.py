# Check impact of feature size
# Include a table that shows max number of features that we take. 500-2000 boundary makes sense for features
# In final report present results in a more concise way
# Create separate tables or graphs between same features for showing results not like the current spaghetti way.
import pandas as pd

import FeatureExtractors
import classification
import utility
import visualization

# data = pd.read_csv('Tweets.csv')
# data = utility.drop_columns(data)
# data = FeatureExtractors.preprocessing(data, 0, 2, 'clean_tweet')
# data = utility.data_balancer(data, True, True, True)
# results = utility.create_results()
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "clean_tweet")
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 1', report,results)
#
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "clean_tweet",max_features=4000)
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 2', report,results)
# visualization.display_metrics(results)


# By uncommenting the code above, you can compare different max feature values in the feature extraction process.
# The code performs two experiments: one with max_features=500 and one with max_features=1500.
# Each experiment includes feature extraction, hyperparameter optimization, and classification using SVM.
# Adjust the values of max_features and run the code to compare the evaluation metrics for different feature sizes.
# Uncommenting this code will display the metrics using the display_metrics function.

# data = pd.read_csv('Tweets.csv')
# data = utility.drop_columns(data)
# data = FeatureExtractors.preprocessing(data, 0, 0, 'no_stemming')
# data = utility.data_balancer(data, True, True, True)
# results = utility.create_results()
#
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "no_stemming",max_features=4000)
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 1', report,results)
#
# data = FeatureExtractors.preprocessing(data, 0, 2, 'porter_stemmer')
# data = utility.data_balancer(data, True, False, False)
#
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "porter_stemmer",max_features=4000)
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 2', report,results)
# visualization.display_metrics(results)

# By uncommenting the code above, you can test the effects of using stemming versus not using it in the text preprocessing.
# The code performs two experiments: one without stemming ('no_stemming') and one with Porter stemming ('porter_stemmer').
# Each experiment includes feature extraction, hyperparameter optimization, and classification using SVM.
# Adjust the parameters and run the code to compare the evaluation metrics for the two experiments.
# Uncommenting this code will display the metrics using the display_metrics function.

# data = pd.read_csv('Tweets.csv')
# data = utility.drop_columns(data)
# data = FeatureExtractors.preprocessing(data, 0, 2, 'clean_tweet')
# data = utility.data_balancer(data, True, True, True)
# results = utility.create_results()
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "clean_tweet")
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 1', report,results)
#
# train_vectors, test_vectors, train_labels, test_labels = utility.feature_extractor(data, "clean_tweet",max_features=4000)
# C, kernel, gamma = utility.perform_hyperparameter_optimization(train_vectors, test_vectors, train_labels, test_labels, 100)
# report = classification.svm_classification(train_vectors, test_vectors, train_labels,
#                                                              test_labels, C, kernel, gamma)
# results = utility.fill_results('Dataset 2', report,results)
# visualization.display_metrics(results)

# By uncommenting the code above, you can test the effects of using the max_features parameter in the tfidf_vectorize function.
# The results dictionary contains sample evaluation metrics for two datasets.
# By passing different values for the max_features parameter in the tfidf_vectorize function, you can observe how the number of features affects the evaluation metrics.
# Adjust the value of max_features and run the code to see the corresponding changes in precision, recall, and F1-score.


# results = {
#     "Dataset 1": {
#         "Precision": {"Positive": 0.93, "Negative": 0.95},
#         "Recall": {"Positive": 0.95, "Negative": 0.93},
#         "F1-Score": {"Positive": 0.94, "Negative": 0.94},
#     },
#     "Dataset 2": {
#         "Precision": {"Positive": 0.94, "Negative": 0.95},
#         "Recall": {"Positive": 0.95, "Negative": 0.95},
#         "F1-Score": {"Positive": 0.95, "Negative": 0.95},
#     },
# }
# visualization.display_metrics(results)