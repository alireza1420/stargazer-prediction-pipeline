
#Caroline script

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/Users/carolineessehorn/Downloads/github_repo_features.csv')

X = df[['forks', 'open_issues', 'size', 'has_wiki', 'has_projects', 'has_downloads', 'is_fork', 'archived',
        'subscribers_count', 'contributors_count', 'commits_count', 'readme_size']]

#convert boolean columns to int
for col in ['has_wiki', 'has_projects', 'has_downloads', 'is_fork', 'archived']:
    X.loc[:, col] = X[col].astype(int)

y = df['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#now performing grid search, so that the best k is to be used
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsRegressor()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

#best parameters and score
best_k = grid.best_params_['n_neighbors']
best_score = grid.best_score_
print("The optimal k is", best_k)

#fitting final model with optimal k
best_knn = KNeighborsRegressor(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

joblib.dump(best_knn, 'knn_model.pkl')

predictions = best_knn.predict(X_test_scaled)

print("R2 score is", r2_score(y_test, predictions))
