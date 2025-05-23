from celery import Celery
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'

# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Load the saved model
pipeline = joblib.load("best_model.pkl")


def transform(df):
    date = datetime(2025, 5, 1)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None)
    df["updated_at"] = pd.to_datetime(df["updated_at"]).dt.tz_localize(None)
    df["pushed_at"] = pd.to_datetime(df["pushed_at"]).dt.tz_localize(None)
    df["project_age"] = (date - df["created_at"]).dt.days
    df["days_since_update"] = (date - df["updated_at"]).dt.days
    df["days_since_push"] = (date - df["pushed_at"]).dt.days
    df["license"] = df["license"].fillna("None")
    df["language"] = df["language"].fillna("Unknown")
    df["forks_per_day"] = df["forks"] / (df["project_age"] + 1)
    df["issues_per_day"] = df["open_issues"] / (df["project_age"] + 1)
    df["update_rate"] = 1 / (1 + df["days_since_update"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    features = [
        'forks', 'open_issues',
       'size', 'has_wiki', 'has_projects', 'has_downloads',
       'archived', 'language', 'license',
       'has_description', 'has_homepage', 'topic_count',
       'has_discussions', 'is_template',
       'subscribers_count', 'contributors_count', 'commits_count',
       'readme_size', 'project_age', 'days_since_push',
       'forks_per_day', 'issues_per_day'
    ]
    return df[features]


@celery.task
def predict_repos_task(repos):
    results = []
    for repo in repos:
        repo_df = pd.DataFrame([repo])
        X_sample = transform(repo_df)
        if X_sample is not None:
            y_pred = pipeline.predict(X_sample)[0]
            results.append({
                'repository': repo['full_name'],
                'language': repo['language'],
                'license': repo.get('license', 'None'),
                'created_at': repo['created_at'],
                'actual_stars': repo.get('stargazers_count', 'N/A'),
                'predicted_stars': round(y_pred),
                'forks': repo['forks'],
                'watchers': repo['watchers'],
                'commits': repo.get('commits_count', 'N/A')
            })

    # Sort by predicted stars in descending order
    results.sort(key=lambda x: x['predicted_stars'], reverse=True)
    return results


def predict_repos(repos):
    task = predict_repos_task.delay(repos)
    return task.get(timeout=30)


@celery.task
def predict_sample_task():
    sample_repo = {
        'name': 'ml-web-app',
        'full_name': 'data-scientist/ml-web-app',
        'created_at': '2023-05-10T08:00:00Z',
        'updated_at': '2023-11-15T14:25:00Z',
        'pushed_at': '2023-11-15T14:30:00Z',
        'language': 'Python',
        'license': 'mit',
        'forks': 87,
        'watchers': 420,
        'open_issues': 12,
        'size': 3500,
        'has_wiki': True,
        'has_projects': False,
        'has_downloads': True,
        'is_fork': False,
        'archived': False,
        'subscribers_count': 150,
        'readme_size': 1024,
        'commits_count': 85,
        'contributors_count': 12
    }
    sample_repo_df = pd.DataFrame([sample_repo])
    X_sample = transform(sample_repo_df)
    if X_sample is not None:
        y_pred = pipeline.predict(X_sample)[0]
        return {
            'repository': sample_repo['full_name'],
            'language': sample_repo['language'],
            'license': sample_repo['license'],
            'created_at': sample_repo['created_at'],
            'predicted_stars': round(y_pred),
            'features': X_sample.columns.tolist()
        }
    else:
        return {'error': 'Prediction failed due to preprocessing error'}


def predict_sample():
    task = predict_sample_task.delay()
    return task.get(timeout=10)