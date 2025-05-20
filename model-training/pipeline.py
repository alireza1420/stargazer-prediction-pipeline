import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os   


class GitHubRepoPreprocessor:
    """Custom preprocessing pipeline for GitHub repository metadata."""
    
    def __init__(self, reference_date=None):
        self.date = reference_date or datetime(2025, 5, 17)
        self.features = [
            'forks', 'open_issues',
       'size', 'has_wiki', 'has_projects', 'has_downloads', 
       'archived', 'language', 'license', 
       'has_description', 'has_homepage', 'topic_count',
       'has_discussions', 'is_template', 
       'subscribers_count', 'contributors_count', 'commits_count',
       'readme_size', 'project_age', 'days_since_push',
       'forks_per_day', 'issues_per_day'
        ] # delete 'watchers', 'is_fork', 'allow_forking', 'visibility', 'days_since_update' and 'update_rate'
        self.numeric_features = [
            'forks', 'open_issues',
            'size', 'topic_count', 'subscribers_count', 
            'contributors_count', 'commits_count', 'readme_size',
            'project_age', 'days_since_push',
            'forks_per_day', 'issues_per_day'
        ]
        self.categorical_features = ["language", "license"]
        self.column_transformer = None

    def transform(self, df):
        # Parse and normalize time-related features
        df["project_age"] = (self.date.date() - df["created_at"].dt.date).apply(lambda x: x.days)
        df["days_since_update"] = (self.date.date() - df["updated_at"].dt.date).apply(lambda x: x.days)
        df["days_since_push"] = (self.date.date() - df["pushed_at"].dt.date).apply(lambda x: x.days)

        # Handle missing values
        df["license"] = df["license"].fillna("None")
        df["language"] = df["language"].fillna("Unknown")

        # Derived rate-based features
        df["forks_per_day"] = df["forks"] / (df["project_age"] + 1)
        df["issues_per_day"] = df["open_issues"] / (df["project_age"] + 1)
        df["update_rate"] = 1 / (1 + df["days_since_update"])
        
        return df[self.features], df["stars"]

    def get_preprocessor(self):
        """Construct and return a fitted ColumnTransformer"""
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
            ]
        )
        return self.column_transformer


def train_and_evaluate_models_cv(df, save_path="best_model_cv.pkl", n_splits=5):
    """Train multiple regressors and save the best performing one."""
    preprocessor = GitHubRepoPreprocessor()
    X, y = preprocessor.transform(df)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocessing pipeline
    processor = preprocessor.get_preprocessor()

    # Candidate models
    models = {
        # Linear Models
        'Linear Regression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=0.5)': Ridge(alpha=0.5),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'ElasticNet (α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.5),
        
        # Tree-based Models
        'Decision Tree (max_depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (max_depth=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        
        # Ensemble Methods
        'Random Forest (n=50)': RandomForestRegressor(n_estimators=50, random_state=42),
        'Random Forest (n=100)': RandomForestRegressor(n_estimators=100, random_state=42),
        'Random Forest (n=200)': RandomForestRegressor(n_estimators=200, random_state=42),
        'Random Forest (n=300)': RandomForestRegressor(n_estimators=300, random_state=42),
        'Gradient Boosting (n=100)': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting (n=200)': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42),
        'Extra Trees': ExtraTreesRegressor(random_state=42),
        'Bagging Regressor': BaggingRegressor(random_state=42),
        
        # SVM
        'SVR (linear)': SVR(kernel='linear'),
        'SVR (rbf)': SVR(kernel='rbf'),
        'SVR (poly)': SVR(kernel='poly'),
        
        # Neighbors
        'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10),
        
        # Neural Networks
        'MLP (1 layer)': MLPRegressor(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),
        'MLP (2 layers)': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
        'MLP (3 layers)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42),
        'MLP (4 layers)': MLPRegressor(hidden_layer_sizes=(128, 64, 32, 16), max_iter=1000, random_state=42),
        
        # Advanced Gradient Boosting
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }

    results = []
    best_cv_score = -np.inf
    best_model = None
    best_pipeline = None
    test_r2_best = -np.inf


    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", processor),
                ("regressor", model)
            ])

            # Cross-validation on training set
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Select best model based on CV mean
            if cv_mean > best_cv_score:
                best_cv_score = cv_mean
                best_model = model

            results.append({
                'Model': name,
                'CV R2 Mean': cv_mean,
                'CV R2 Std': cv_std,
            })

            print(f"{name:<25} | CV R2: {cv_mean:.4f} ± {cv_std:.4f}")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue

    if best_model is not None:
        final_pipeline = Pipeline([
            ("preprocessor", processor),
            ("regressor", best_model)
        ])
        final_pipeline.fit(X_train, y_train)
        y_test_pred = final_pipeline.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"\n=== Best Model: {best_model.__class__.__name__} ===")
        print(f"Test R2 Score: {test_r2:.4f}")

        joblib.dump(final_pipeline, save_path)
        print(f"Best model saved to: {os.path.abspath(save_path)}")

        # Save the best model
        test_r2_best = test_r2
        best_pipeline = final_pipeline

    # Return summary
    results_df = pd.DataFrame(results).sort_values(by='CV R2 Mean', ascending=False)
    return results_df, best_pipeline, test_r2_best

import matplotlib.pyplot as plt
def visualize_results_cv(results_df):
    plt.figure(figsize=(12, 8))
    
    # Sort by R2 score and plot
    results_df = results_df.sort_values(by='CV R2 Mean')

    bars = plt.barh(results_df['Model'], results_df['CV R2 Mean'], color='skyblue')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                va='center', ha='left', fontsize=9)
    
    # Customize plot
    plt.xlabel('R2 Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and show
    plt.savefig('model_comparison_cv.png', dpi=300, bbox_inches='tight')
    # plt.show()

# load data
df = pd.read_csv('github_repo_features_new.csv', parse_dates=["created_at", "updated_at", "pushed_at"])
results_df_cv, best_pipeline, test_r2_best = train_and_evaluate_models_cv(df, save_path="best_model_cv.pkl")
# Visualize results
visualize_results_cv(results_df_cv)