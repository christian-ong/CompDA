import scipy . linalg as lng # linear algebra from scipy library
import numpy as np # numpy library
from scipy . spatial import distance
from sklearn.linear_model import Lasso, ElasticNet



######################################## OLS ########################################
def ols_numerical(X, y):
    X= np.column_stack((np.ones(len(y)), X)) #intercept
    betas = lng.lstsq(X, y)[0]
    prediction = X @ betas
    return prediction


def ols_analytical(X, y):
    X= np.column_stack((np.ones(len(y)), X)) #intercept
    betas = lng.pinv(X.T @ X) @ X.T @ y
    prediction = X @ betas
    return prediction



######################################## KNN ########################################
def knn(X, y, K):
    dist_matrix = distance.cdist(X, X)  # Compute all pairwise distances
    idx = np.argsort(dist_matrix, axis=1)[:, 1:K+1]  # Get nearest neighbors
    yhat = np.mean(y[idx], axis=1)  # Compute mean for KNN
    return yhat


def weighted_knn(X, y, K):
    n = len(y)
    yhat = np.zeros(n)

    for i in range(n):
        dist = np.zeros(n)
        for j in range(n):
            dist[j] = distance.euclidean(X[i, :], X[j, :])
        idx = np.argsort(dist)
        yhat[i] = np.sum(y[idx[0:K]] / dist[idx[0:K]]) / np.sum(1 / dist[idx[0:K]])
    return yhat

######################################## logistic regression ########################################
def logistic_regression_analytical(X, y):
    X = np.column_stack((np.ones(len(y)), X))  # Add intercept
    betas = lng.lstsq(X, y)[0]
    yhat = 1 / (1 + np.exp(-X @ betas))  # Final predictions
    return yhat


######################################## Ridge regression ########################################
def ridge_regression(X, y, lmda):
    X= np.column_stack((np.ones(len(y)), X)) #intercept
    n, p = X.shape
    betas = np.zeros(p)
    
    betas = lng.inv(X.T @ X + lmda * np.eye(p)) @ X.T @ y
    prediction = X @ betas
    return prediction

######################################## Lasso regression ########################################
def lasso_regression(X, y, lmda):
    model = Lasso(alpha=lmda, fit_intercept=True)
    model.fit(X, y)
    return model.predict(X)

######################################## Elastic Net regression ########################################
def elastic_net_regression(X, y, lmda, alpha):
    model = ElasticNet(alpha=lmda, l1_ratio=alpha, fit_intercept=True)
    model.fit(X, y)
    return model.predict(X)
