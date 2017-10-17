import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Compute Parameterized SVM with grid search
# make sure that a high score is better than a low score! If you use accuracy, a high accuracy is better than a low
# one. If you use Error (e.g. MSE) make sure it is negative (negative MSE)!

def compute(X, y, mask_sub_features, X_test, y_test):
    param_grid = [{'C': np.logspace(-1, 1, 6), 'gamma': np.logspace(-1, 1, 6)}]
    clf = svm.SVC()  # SVR for regression, SVC for classification
    grid = GridSearchCV(clf, param_grid, cv=None, n_jobs=-1, scoring='neg_mean_squared_error', pre_dispatch=8)
    grid.fit(X, y)
    y_pred = grid.predict(X_test[:, mask_sub_features])
    score = accuracy_score(y_test, y_pred)
    print score
    return score
    # print grid.cv_results_
    # print (grid.grid_scores_)
    print(grid.best_score_)
    # print(grid.best_params_)
    # return grid.best_score_
