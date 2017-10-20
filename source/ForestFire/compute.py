# Imports
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# make sure that a high score is better than a low score! If you use accuracy, a high accuracy is better than a low
# one. If you use Error (e.g. MSE) make sure it is negative (negative MSE)!


def compute(X_train, y_train, mask_sub_features, X_test, y_test):
    """Computes a new dataset for the Random Forest with the underlying machine learning algorithm.

    Configure your machine learning algorithm here. 
    Make imports at the top of the file.

    Arguments:

        * X_train {np.array} -- feature training data
        * y_train {np.array} -- result training data
        * mask_sub_features {np.array} -- feature set, only part of all features
        * X_test {np.array} -- result training data
        * y_test {np.array} -- result test data

    Returns:
        score {np.float64} -- score of the selected feature set
    """

    ### insert your own machine learning algorithm ###
    param_grid = [{'C': np.logspace(-1, 1, 6), 'gamma': np.logspace(-1, 1, 6)}]
    clf = svm.SVC()  # SVR for regression, SVC for classification
    grid = GridSearchCV(clf, param_grid, cv=None, n_jobs=-1, scoring='neg_mean_squared_error', pre_dispatch=8)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test[:, mask_sub_features])

    ### store the result in score ###
    score = accuracy_score(y_test, y_pred)
    # print score
    return score
    # print grid.cv_results_
    # print (grid.grid_scores_)
    # print(grid.best_score_)
    # print(grid.best_params_)
    # return grid.best_score_
