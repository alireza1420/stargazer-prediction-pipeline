import matplotlib.pyplot as plt
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
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from sklearn.linear_model import (HuberRegressor, QuantileRegressor, 
                                BayesianRidge, ARDRegression, 
                                PassiveAggressiveRegressor, TheilSenRegressor,
                                OrthogonalMatchingPursuit, RANSACRegressor)
from sklearn.ensemble import (HistGradientBoostingRegressor, 
                            StackingRegressor, VotingRegressor)
from sklearn.neighbors import RadiusNeighborsRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor  # For probabilistic forecasting
import joblib
import os




# load data
df = pd.read_csv('github_repo_features_new.csv', parse_dates=["created_at", "updated_at", "pushed_at"])

# Features of the dataset
print(df.columns)
# Number of Features
len(df.columns)


# Parse and normalize time-related features
date = pd.to_datetime("2025-5-17")
df["project_age"] = (date.date() - df["created_at"].dt.date).apply(lambda x: x.days)
df["days_since_update"] = (date.date() - df["updated_at"].dt.date).apply(lambda x: x.days)
df["days_since_push"] = (date.date() - df["pushed_at"].dt.date).apply(lambda x: x.days)


# Handle missing values
df["license"] = df["license"].fillna("None")
df["language"] = df["language"].fillna("Unknown")

# Derived rate-based features
df["forks_per_day"] = df["forks"] / (df["project_age"] + 1)
df["issues_per_day"] = df["open_issues"] / (df["project_age"] + 1)
df["update_rate"] = 1 / (1 + df["days_since_update"])




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


def train_and_evaluate_models(df, save_path="best_model_cv.pkl"):
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
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Ridge (α=0.5)': Ridge(alpha=0.5),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=5.0)': Ridge(alpha=5.0),
        'Lasso (α=0.001)': Lasso(alpha=0.001),
        'Lasso (α=0.01)': Lasso(alpha=0.01),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'ElasticNet (α=0.001)': ElasticNet(alpha=0.001, l1_ratio=0.5),
        'ElasticNet (α=0.01)': ElasticNet(alpha=0.01, l1_ratio=0.3),
        'ElasticNet (α=0.1)': ElasticNet(alpha=0.1, l1_ratio=0.7),
        'Huber Regressor (ϵ=1.35)': HuberRegressor(epsilon=1.35),
        'Huber Regressor (ϵ=2.0)': HuberRegressor(epsilon=2.0),
        'Quantile Regressor (α=0.5)': QuantileRegressor(quantile=0.5),
        'Quantile Regressor (α=0.9)': QuantileRegressor(quantile=0.9),
        'Bayesian Ridge': BayesianRidge(),
        'ARD Regression': ARDRegression(),
        'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=42),
        'Theil-Sen Regressor': TheilSenRegressor(random_state=42),
        'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
        'RANSAC Regressor': RANSACRegressor(random_state=42),
        
        # Tree-based Models
        'Decision Tree (max_depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
        'Decision Tree (max_depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (max_depth=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Decision Tree (max_depth=None)': DecisionTreeRegressor(max_depth=None, random_state=42),
        'Decision Tree (min_samples_leaf=5)': DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
        'Decision Tree (min_samples_split=10)': DecisionTreeRegressor(min_samples_split=10, random_state=42),
        'Decision Tree (max_features=sqrt)': DecisionTreeRegressor(max_features='sqrt', random_state=42),
        'Decision Tree (max_features=log2)': DecisionTreeRegressor(max_features='log2', random_state=42),
        
        # Ensemble Methods
        'Random Forest (n=50)': RandomForestRegressor(n_estimators=50, random_state=42),
        'Random Forest (n=100)': RandomForestRegressor(n_estimators=100, random_state=42),
        'Random Forest (n=200)': RandomForestRegressor(n_estimators=200, random_state=42),
        'Random Forest (n=300)': RandomForestRegressor(n_estimators=300, random_state=42),
        'Random Forest (max_depth=5)': RandomForestRegressor(max_depth=5, random_state=42),
        'Random Forest (max_depth=10)': RandomForestRegressor(max_depth=10, random_state=42),
        'Random Forest (max_features=sqrt)': RandomForestRegressor(max_features='sqrt', random_state=42),
        'Random Forest (max_features=log2)': RandomForestRegressor(max_features='log2', random_state=42),
        'Random Forest (min_samples_leaf=5)': RandomForestRegressor(min_samples_leaf=5, random_state=42),
        'Random Forest (bootstrap=False)': RandomForestRegressor(bootstrap=False, random_state=42),
        
        
        # Gradient Boosting
        'Gradient Boosting (n=50)': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting (n=100)': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting (n=200)': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'Gradient Boosting (learning_rate=0.01)': GradientBoostingRegressor(learning_rate=0.01, random_state=42),
        'Gradient Boosting (learning_rate=0.1)': GradientBoostingRegressor(learning_rate=0.1, random_state=42),
        'Gradient Boosting (max_depth=3)': GradientBoostingRegressor(max_depth=3, random_state=42),
        'Gradient Boosting (max_depth=5)': GradientBoostingRegressor(max_depth=5, random_state=42),
        'Gradient Boosting (max_depth=7)': GradientBoostingRegressor(max_depth=7, random_state=42),
        'Gradient Boosting (max_depth=9)': GradientBoostingRegressor(max_depth=9, random_state=42),
        'Gradient Boosting (subsample=0.5)': GradientBoostingRegressor(subsample=0.5, random_state=42),
        'Gradient Boosting (max_features=sqrt)': GradientBoostingRegressor(max_features='sqrt', random_state=42),
        'Gradient Boosting (loss=absolute)': GradientBoostingRegressor(loss='absolute_error', random_state=42),
        'Gradient Boosting (loss=huber)': GradientBoostingRegressor(loss='huber', random_state=42),
        
        #AdaBoost
        'AdaBoost (DT base)': AdaBoostRegressor(random_state=42),
        'AdaBoost (n=50)': AdaBoostRegressor(n_estimators=50, random_state=42),
        'AdaBoost (n=100)': AdaBoostRegressor(n_estimators=100, random_state=42),
        'AdaBoost (n=200)': AdaBoostRegressor(n_estimators=200, random_state=42),
        'AdaBoost (learning_rate=0.5)': AdaBoostRegressor(learning_rate=0.5, random_state=42),
        'AdaBoost (learning_rate=1.0)': AdaBoostRegressor(learning_rate=1.0, random_state=42),
        
        
        #Extra Tree
        'Extra Trees (n=50)': ExtraTreesRegressor(n_estimators=50, random_state=42),
        'Extra Trees (n=100)': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'Extra Trees (n=200)': ExtraTreesRegressor(n_estimators=200, random_state=42),
        'Extra Trees (n=250)': ExtraTreesRegressor(n_estimators=250, random_state=42),
        'Extra Trees (n=300)': ExtraTreesRegressor(n_estimators=300, random_state=42),
        'Extra Trees (max_depth=5)': ExtraTreesRegressor(max_depth=5, random_state=42),
        'Extra Trees (max_depth=10)': ExtraTreesRegressor(max_depth=10, random_state=42),
        'Extra Trees (max_features=sqrt)': ExtraTreesRegressor(max_features='sqrt', random_state=42),
        'Extra Trees (max_features=log2)': ExtraTreesRegressor(max_features='log2', random_state=42),
        'Extra Trees (bootstrap=True)': ExtraTreesRegressor(bootstrap=True, random_state=42),
        
        
        #Bagging
        'Bagging (DT base)': BaggingRegressor(random_state=42),
        'Bagging (n=50)': BaggingRegressor(n_estimators=50, random_state=42),
        'Bagging (n=100)': BaggingRegressor(n_estimators=100, random_state=42),
        'Bagging (n=200)': BaggingRegressor(n_estimators=200, random_state=42),
        'Bagging (max_samples=0.5)': BaggingRegressor(max_samples=0.5, random_state=42),
        'Bagging (max_features=0.5)': BaggingRegressor(max_features=0.5, random_state=42),
        'Bagging (bootstrap=False)': BaggingRegressor(bootstrap=False, random_state=42),
        
        
        # SVM
        'SVR (linear)': SVR(kernel='linear'),
        'SVR (rbf)': SVR(kernel='rbf'),
        'SVR (poly)': SVR(kernel='poly'),
        'SVR (sigmoid)': SVR(kernel='sigmoid'),
        'SVR (C=0.1)': SVR(C=0.1),
        'SVR (C=1.0)': SVR(C=1.0),
        'SVR (C=10)': SVR(C=10),
        'SVR (epsilon=0.1)': SVR(epsilon=0.1),
        'SVR (epsilon=0.5)': SVR(epsilon=0.5),
        'NuSVR (nu=0.1)': NuSVR(nu=0.1),
        'NuSVR (nu=0.5)': NuSVR(nu=0.5),
        'LinearSVR (epsilon=0.0)': LinearSVR(epsilon=0.0, random_state=42),
        'LinearSVR (epsilon=1.0)': LinearSVR(epsilon=1.0, random_state=42),
        
        # Neighbors
        'KNN (k=3)': KNeighborsRegressor(n_neighbors=3),
        'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
        'KNN (k=10)': KNeighborsRegressor(n_neighbors=10),
        'KNN (k=20)': KNeighborsRegressor(n_neighbors=20),
        'KNN (weights=uniform)': KNeighborsRegressor(weights='uniform'),
        'KNN (weights=distance)': KNeighborsRegressor(weights='distance'),
        'KNN (p=1)': KNeighborsRegressor(p=1),  # Manhattan distance
        'KNN (p=2)': KNeighborsRegressor(p=2),  # Euclidean distance
        
        # Neural Networks
        'MLP (1 layer)': MLPRegressor(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),
        'MLP (2 layers)': MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42),
        'MLP (3 layers)': MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42),
        'MLP (4 layers)': MLPRegressor(hidden_layer_sizes=(128, 64, 32, 16), max_iter=1000, random_state=42),
        'MLP (adam)': MLPRegressor(solver='adam', random_state=42),
        'MLP (lbfgs)': MLPRegressor(solver='lbfgs', random_state=42),
        'MLP (sgd)': MLPRegressor(solver='sgd', random_state=42),
        'MLP (relu)': MLPRegressor(activation='relu', random_state=42),
        'MLP (tanh)': MLPRegressor(activation='tanh', random_state=42),
        'MLP (logistic)': MLPRegressor(activation='logistic', random_state=42),
        'MLP (alpha=0.0001)': MLPRegressor(alpha=0.0001, random_state=42),
        'MLP (alpha=0.001)': MLPRegressor(alpha=0.001, random_state=42),
        'MLP (batch_size=32)': MLPRegressor(batch_size=32, random_state=42),
        'MLP (batch_size=64)': MLPRegressor(batch_size=64, random_state=42),
        
        
        # Advanced Gradient Boosting (LightGBM)
        'LightGBM (n=100)': LGBMRegressor(n_estimators=100, random_state=42),
        'LightGBM (n=200)': LGBMRegressor(n_estimators=200, random_state=42),
        'LightGBM (max_depth=3)': LGBMRegressor(max_depth=3, random_state=42),
        'LightGBM (max_depth=6)': LGBMRegressor(max_depth=6, random_state=42),
        'LightGBM (learning_rate=0.001)': LGBMRegressor(learning_rate=0.001, random_state=42),
        'LightGBM (learning_rate=0.01)': LGBMRegressor(learning_rate=0.01, random_state=42),
        'LightGBM (learning_rate=0.05)': LGBMRegressor(learning_rate=0.05, random_state=42),
        'LightGBM (learning_rate=0.1)': LGBMRegressor(learning_rate=0.1, random_state=42),
        'LightGBM (num_leaves=31)': LGBMRegressor(num_leaves=31, random_state=42),
        'LightGBM (num_leaves=63)': LGBMRegressor(num_leaves=63, random_state=42),
        'LightGBM (subsample=0.5)': LGBMRegressor(subsample=0.5, random_state=42),
        'LightGBM (colsample_bytree=0.5)': LGBMRegressor(colsample_bytree=0.5, random_state=42),
        'LightGBM (reg_alpha=0)': LGBMRegressor(reg_alpha=0, random_state=42),
        'LightGBM (reg_alpha=1)': LGBMRegressor(reg_alpha=1, random_state=42),
        'LightGBM (reg_lambda=0)': LGBMRegressor(reg_lambda=0, random_state=42),
        'LightGBM (reg_lambda=1)': LGBMRegressor(reg_lambda=1, random_state=42),
        'LightGBM (boosting_type=gbdt)': LGBMRegressor(boosting_type='gbdt', random_state=42),
        'LightGBM (boosting_type=dart)': LGBMRegressor(boosting_type='dart', random_state=42),
        'LightGBM (boosting_type=goss)': LGBMRegressor(boosting_type='goss', random_state=42),
        'LightGBM (objective=regression)': LGBMRegressor(objective='regression', random_state=42),
        'LightGBM (objective=mae)': LGBMRegressor(objective='mae', random_state=42),
        
        # Advanced Gradient Boosting (CatBOOST)
        'CatBoost (n=100)': CatBoostRegressor(iterations=100, random_state=42, verbose=0),
        'CatBoost (n=200)': CatBoostRegressor(iterations=200, random_state=42, verbose=0),
        'CatBoost (depth=3)': CatBoostRegressor(depth=3, random_state=42, verbose=0),
        'CatBoost (depth=6)': CatBoostRegressor(depth=6, random_state=42, verbose=0),
        'CatBoost (learning_rate=0.001)': CatBoostRegressor(learning_rate=0.001, random_state=42, verbose=0),
        'CatBoost (learning_rate=0.01)': CatBoostRegressor(learning_rate=0.01, random_state=42, verbose=0),
        'CatBoost (learning_rate=0.05)': CatBoostRegressor(learning_rate=0.05, random_state=42, verbose=0),
        'CatBoost (learning_rate=0.1)': CatBoostRegressor(learning_rate=0.1, random_state=42, verbose=0),
        'CatBoost (l2_leaf_reg=1)': CatBoostRegressor(l2_leaf_reg=1, random_state=42, verbose=0),
        'CatBoost (l2_leaf_reg=3)': CatBoostRegressor(l2_leaf_reg=3, random_state=42, verbose=0),
        'CatBoost (border_count=32)': CatBoostRegressor(border_count=32, random_state=42, verbose=0),
        'CatBoost (border_count=128)': CatBoostRegressor(border_count=128, random_state=42, verbose=0),
        'CatBoost (grow_policy=SymmetricTree)': CatBoostRegressor(grow_policy='SymmetricTree', random_state=42, verbose=0),
        'CatBoost (grow_policy=Lossguide)': CatBoostRegressor(grow_policy='Lossguide', random_state=42, verbose=0),
        'CatBoost (loss_function=RMSE)': CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=0),
        'CatBoost (loss_function=MAE)': CatBoostRegressor(loss_function='MAE', random_state=42, verbose=0),
        'CatBoost (one_hot_max_size=2)': CatBoostRegressor(one_hot_max_size=2, random_state=42, verbose=0),
        'CatBoost (one_hot_max_size=10)': CatBoostRegressor(one_hot_max_size=10, random_state=42, verbose=0),
        
    }

    results = []
    best_model = None
    best_score = -np.inf


    for name, model in models.items():
        try:
            pipeline = Pipeline([
                ("preprocessor", processor),
                ("regressor", model)
            ])

            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)

            # Store results
            model_params = str(model.get_params())
            results.append({
                'Model': name,
                'Parameters': model_params,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })

            # Save the best model
            if r2 > best_score:
                best_score = r2
                best_model = Pipeline([
                ("preprocessor", processor),
                ("regressor", model)
            ])
                
            print(f"{name: <30} | R2: {r2:.4f} | RMSE: {rmse:.2f}")
            # print(f"{name: <30} | R2: {r2:.4f} | RMSE: {rmse:.2f} | {best_model: <30} ({best_score: <30})")
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='R2', ascending=False)

    # Display all results
    print("\n=== All Model Results ===")
    print(results_df.to_string())

    if best_model:
        joblib.dump(best_model, save_path)
        #print(f"\n Best model {best_model: <30} ({best_score: <30})")
        print(f"\n Best model saved to: {os.path.abspath(save_path)}")
    
    return results_df



def visualize_results(results_df):
    plt.figure(figsize=(12, 8))

    # Sort by R2 score in descending order
    results_df = results_df.sort_values(by='R2', ascending=False)

    # Select the top 10 models
    top_20_df = results_df.head(20)

    # Plot the top 20
    bars = plt.barh(top_20_df['Model'], top_20_df['R2'], color='skyblue')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 va='center', ha='left', fontsize=9)

    # Customize plot
    plt.xlabel('R2 Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Top 20 Model Performance Comparison (R2 Score)', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save
    plt.savefig('top_20_model_comparison.png', dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    # load data
    df = pd.read_csv('github_repo_features_new.csv', parse_dates=["created_at", "updated_at", "pushed_at"])
    results_df = train_and_evaluate_models(df, save_path="best_model.pkl")

    #Number of experments have been done
    len(results_df)

    visualize_results(results_df)



