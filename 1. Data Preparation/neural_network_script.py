#Caroline's script

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv('/Users/carolineessehorn/Downloads/github_repo_features.csv')

X = df[['forks', 'watchers', 'open_issues', 'size', 'has_wiki', 'has_projects', 'has_downloads', 'is_fork', 'archived',
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

#define the model
model = Sequential()
model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

#fit the model to the dataset
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.2)

model.save('neural_network_model.h5')

predictions = model.predict(X_test)

#evaluate the model
_, R2_score = model.evaluate(X_test, y_test)
print('R2 score:', r2_score(y_test, predictions))

