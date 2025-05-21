
#Caroline's script
#this is based on Sckitlearn framework instead of tensorflow obs

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv('/Users/carolineessehorn/Downloads/github_repo_features.csv')

X = df[['forks', 'open_issues', 'size', 'has_wiki', 'has_projects', 'has_downloads', 'is_fork', 'archived',
        'subscribers_count', 'contributors_count', 'commits_count', 'readme_size']]

for col in ['has_wiki', 'has_projects', 'has_downloads', 'is_fork', 'archived']:
    X.loc[:, col] = X[col].astype(int)

y = df['stars']

#data splitting into training and testing data (80/20) using random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

#another way to save the model, when not using tensorflow framework:
joblib.dump(model, 'linear_regression_model.pkl')

predictions = model.predict(X_test)

print('R2 score:', r2_score(y_test, predictions))
