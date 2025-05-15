import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import joblib

class GitHubRepoPreprocessor:
    def __init__(self, reference_date=None):
        self.date = reference_date or datetime(2025, 5, 1)
        self.numeric_features = [
            'forks', 'open_issues', 'size', 'subscribers_count', 
            'contributors_count', 'commits_count', 'readme_size',
            'project_age', 'days_since_update', 'days_since_push',
            'forks_per_day', 'issues_per_day', 'update_rate'
        ]
        self.categorical_features = ["language", "license"]
        self.column_transformer = None

    def transform(self, df):
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
        df["updated_at"] = pd.to_datetime(df["updated_at"]).dt.tz_localize(None)
        df["pushed_at"] = pd.to_datetime(df["pushed_at"]).dt.tz_localize(None)

        df["project_age"] = (self.date - df["created_at"]).dt.days
        df["days_since_update"] = (self.date - df["updated_at"]).dt.days
        df["days_since_push"] = (self.date - df["pushed_at"]).dt.days

        df["license"] = df["license"].fillna("None")
        df["language"] = df["language"].fillna("Unknown")

        df["forks_per_day"] = df["forks"] / (df["project_age"] + 1)
        df["issues_per_day"] = df["open_issues"] / (df["project_age"] + 1)
        df["update_rate"] = 1 / (1 + df["days_since_update"])

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df["has_wiki"] = df["has_wiki"].astype(int)
        df["has_projects"] = df["has_projects"].astype(int)
        df["has_downloads"] = df["has_downloads"].astype(int)
        df["is_fork"] = df["is_fork"].astype(int)
        df["archived"] = df["archived"].astype(int)

        selected_features = [
            'forks', 'open_issues', 'size', 'has_wiki', 'has_projects', 'has_downloads',
            'is_fork', 'archived', 'language', 'license',
            'subscribers_count', 'contributors_count', 'commits_count', 'readme_size',
            'project_age', 'days_since_update', 'days_since_push',
            'forks_per_day', 'issues_per_day', 'update_rate'
        ]

        return df[selected_features], df["stars"]

    def get_preprocessor(self):
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
            ],
            remainder='passthrough'
        )
        return self.column_transformer


df = pd.read_csv('/Users/carolineessehorn/Downloads/github_repo_features.csv')
preprocessor = GitHubRepoPreprocessor()
X, y = preprocessor.transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("preprocess", preprocessor.get_preprocessor()),
    ("regressor", DecisionTreeRegressor(random_state=42))
])

param_grid = {
    "regressor__max_depth": [3, 5, 10, None],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation R2 score:", grid.best_score_)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)
print("R2 score on test set:", r2_score(y_test, predictions))

joblib.dump(best_model, "decision_tree_pipeline_model.pkl")

