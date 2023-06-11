import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, \
    classification_report, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def svm_classification(train_vectors, test_vectors, train_labels, test_labels, C, kernel, gamma):
    # Train an SVM classifier on the training set
    classifier = SVC(kernel=kernel, C=C, gamma=gamma, random_state=10)
    classifier.fit(train_vectors, train_labels)

    # Predict the sentiment labels for the test set
    predicted_labels = classifier.predict(test_vectors)

    # Define the scoring metric with weighted average
    scoring = make_scorer(f1_score, average='macro')

    # Perform cross-validation with weighted average
    scores = cross_val_score(classifier, train_vectors, train_labels, cv=5, scoring=scoring)

    # Evaluate the accuracy, precision, and recall of the classifier
    confusion_mat = confusion_matrix(test_labels, predicted_labels)
    print(confusion_mat)
    print(np.mean(scores))
    print(classification_report(test_labels, predicted_labels))
    report = classification_report(test_labels, predicted_labels, output_dict=True)
    return report


def objective(trial, train_vectors, test_vectors, train_labels, test_labels):
    # Define the search space for hyperparameters
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
    gamma = trial.suggest_float('gamma', 0.001, 1.0, log=True)

    # Create the SVM classifier with the sampled hyperparameters
    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(train_vectors, train_labels)

    # Predict the sentiment labels for the test set
    predicted_labels = classifier.predict(test_vectors)
    f1 = f1_score(test_labels, predicted_labels, average='macro')

    # Optimize the objective function (in this case, maximize the f1 score)
    return f1
