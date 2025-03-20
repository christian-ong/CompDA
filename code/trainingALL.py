import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.spatial import distance
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

# Load cleaned data
DATA_PATH = "../../data/cleanedData.csv"
data = pd.read_csv(DATA_PATH)

X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target

# Standardize data before PCA
X_scaled = StandardScaler().fit_transform(X)

# Fit PCA
pca = PCA().fit(X_scaled)

# Compute cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components that explain at least 95% of variance
optimal_n = np.argmax(explained_variance >= 0.95) + 1  # Adding 1 because index starts at 0

X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values
# optimal_n = 41
# Define Models, PIPELINE- standardize and then PCA in the cross-val splitted data to avoid data leakage
# models = {
#     "OLS": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), LinearRegression()),
#     "Ridge": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), Ridge(alpha=1.0)),
#     "Lasso": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), Lasso(alpha=0.1)),
#     "ElasticNet": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), ElasticNet(alpha=0.1, l1_ratio=0.5)),
#     "RandomForest": make_pipeline(PCA(n_components=optimal_n), RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
# }

models = {
    "OLS": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), BaggingRegressor(LinearRegression(), n_estimators=10, random_state=42)),
    "Ridge": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), BaggingRegressor(Ridge(alpha=1.0), n_estimators=10, random_state=42)),
    "Lasso": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), BaggingRegressor(Lasso(alpha=0.1), n_estimators=10, random_state=42)),
    "ElasticNet": make_pipeline(StandardScaler(), PCA(n_components=optimal_n), BaggingRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5), n_estimators=10, random_state=42)),
    "RandomForest": make_pipeline(PCA(n_components=optimal_n), RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))  # No need for bagging here
}

results2 = {}
# LOO Cross-Validation
loo = LeaveOneOut()
kf = KFold(n_splits=5)


for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=loo, scoring='neg_root_mean_squared_error')
    y_pred = -scores
    
    if name in results2:
        results2[name].append(np.mean(y_pred))
    else:
        results2[name] = [np.mean(y_pred)]

# Print Results
print("Model Performance (LOO RMSE):")
for model, score in results2.items():
    print(f"{model}: {score}")