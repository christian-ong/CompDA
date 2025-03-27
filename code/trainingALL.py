import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, accuracy_score
from scipy.spatial import distance
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import functions as f
import os
#print working directory
print(os.getcwd())

# Get directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Build correct path to data
DATA_PATH = os.path.join(DATA_DIR, "case1Data.csv")

data = pd.read_csv(DATA_PATH)

data = f.clean_data(data, fill_type=None, one_hot=False)

X = data.drop(columns=["y"])  # Features (all columns except the first)
y = data['y']  # Target (first column)

categorical_features = ["C_01", "C_03", "C_04", "C_05"]
numeric_features = [col for col in X.columns if col not in categorical_features]

numeric_transform = Pipeline(steps=[
    ("imputer", KNNImputer()),       
    ("scaler", StandardScaler())
])

categorical_transform = Pipeline(steps=[
    ("imputer", KNNImputer()),  # Impute missing categorical values
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transform, numeric_features),         # standardize numeric
        ("cat", categorical_transform, categorical_features)         # pass one-hot columns as-is (or use OneHotEncoder if needed)
    ])
# Define Models (No PCA here, applied inside GridSearch)
models = {
    # "OLS": make_pipeline(preprocessor, PCA(), LinearRegression()),
    # "Ridge": make_pipeline(preprocessor, PCA(), Ridge()),
    # "Lasso": make_pipeline(preprocessor, PCA(), Lasso()),
    "ElasticNet": make_pipeline(preprocessor, PCA(), ElasticNet()),
    "SVR": make_pipeline(preprocessor, PCA(), SVR()),
    "RandomForest": make_pipeline(preprocessor, RandomForestRegressor(random_state=42)),  # No pipeline needed
    "KNN": make_pipeline(preprocessor, PCA(), KNeighborsRegressor())
}

# Define hyperparameter grids
param_grids = {
    # "OLS": {'pca__n_components': [60, 70], 'columntransformer__num__imputer__n_neighbors': [5, 10]},
    # "Ridge": {'pca__n_components': [60, 70], 'ridge__alpha': [0.1, 1], 'columntransformer__num__imputer__n_neighbors': [5, 10]},
    # "Lasso": {'pca__n_components': [30, 37], 'lasso__alpha': [0.1, 1], 'columntransformer__num__imputer__n_neighbors': [5, 10]},
    "ElasticNet": {'pca__n_components': [50, 60, 70, 79], 
                   'elasticnet__alpha': [1e-4, 1e-3, 0.1, 1], 
                   'elasticnet__l1_ratio': [0.1, 0.5, 0.7, 0.9], 
                   'columntransformer__num__imputer__n_neighbors': [3, 5, 10, 15],
                   'columntransformer__cat__imputer__n_neighbors': [3, 5, 10, 15]},

    "SVR": {'pca__n_components': [50, 60, 70, 79], 
            'svr__C': [1e-4, 1e-3, 0.1, 1], 
            'svr__epsilon': [1e-3, 0.1, 1], 
            'svr__kernel': ['linear', 'rbf'], 
            'columntransformer__num__imputer__n_neighbors': [5, 10, 15, 20],
            'columntransformer__cat__imputer__n_neighbors': [3, 5, 10, 15]},

    "RandomForest": {'randomforestregressor__n_estimators': [10, 30, 50, 100],
                     'columntransformer__num__imputer__n_neighbors': [5, 10, 15, 20], 
                     'randomforestregressor__max_depth': [3, 5, 10, 20], 
                     'randomforestregressor__min_samples_split': [2, 5],
                     'columntransformer__cat__imputer__n_neighbors': [3, 5, 10, 15]},

    "KNN": {'pca__n_components': [50, 60, 70, 79], 
            'kneighborsregressor__n_neighbors': [3, 5, 10, 15, 20], 
            'kneighborsregressor__weights': ['uniform', 'distance'], 
            'columntransformer__num__imputer__n_neighbors': [5, 10, 15, 20],
            'columntransformer__cat__imputer__n_neighbors': [3, 5, 10, 15]}
}

# Outer Loop: Leave-One-Out Cross-Validation
loo = LeaveOneOut()
KFcv = KFold(n_splits=5, shuffle=True, random_state=42)
final_rmse_scores = {model: [] for model in models}

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Inner Loop: Hyperparameter Tuning using GridSearchCV (on training data)
    best_params = {}

    for model_name, pipeline in models.items():
        grid_search = GridSearchCV(pipeline, param_grid=param_grids[model_name], cv=KFcv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params[model_name] = grid_search.best_params_

        # Train model with best parameters on full training data
        base_model = models[model_name].set_params(**best_params[model_name])
        # Wrap in bagging if model benefits from it
        if model_name not in ["RandomForest"]:
            estimators = [est for name, est in base_model.steps[:-1]]
            best_model = make_pipeline(
                *estimators,  # all except the final estimator
                BaggingRegressor(base_model.steps[-1][1], n_estimators=10, random_state=42)
            )
        else:
            best_model = base_model

        best_model.fit(X_train, y_train) # use strategy 1 bagging?

        # Predict and calculate RMSE
        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)

        # Store RMSE for this fold
        final_rmse_scores[model_name].append(rmse)

# Compute mean RMSE across all LOO iterations
for model, scores in final_rmse_scores.items():
    print(f"{model}: Mean RMSE across LOO = {np.mean(scores):.4f}")


# Find the best model and its parameters
best_model_name = min(final_rmse_scores, key=lambda model: np.mean(final_rmse_scores[model]))
final_model = models[best_model_name]

# Predict `X_new`
X_new = pd.read_csv("../../data/case1Data_Xnew.csv") #data cleaning?
X_new = f.clean_data(X_new, fill_type=None, one_hot=False)

# Train the best model on the full dataset (since LOO already ensured a fair RMSE estimate)
best_model = final_model.set_params(**best_params[best_model_name])
# bagging for stable predictions
estimators = [est for name, est in best_model.steps[:-1]]
best_model = make_pipeline(
    *estimators,  # all except the final estimator
    BaggingRegressor(best_model.steps[-1][1], n_estimators=10, random_state=42)
)
best_model.fit(X, y)

# final predictions
y_new_pred = best_model.predict(X_new)

#save predictions
pred_path = os.path.join(DATA_DIR, "sample_predictions_s204109_s214601.csv")
np.savetxt(pred_path, y_new_pred, delimiter=",")


#save RMSE
rmse_path = os.path.join(DATA_DIR, "sample_estimatedRMSE_s204109_s214601.csv")
np.savetxt(rmse_path, [np.mean(final_rmse_scores[best_model_name])])