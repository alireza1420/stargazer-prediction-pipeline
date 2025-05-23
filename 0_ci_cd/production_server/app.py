from flask import Flask, render_template, request, redirect, url_for
from workerA import predict_sample, predict_repos
from datetime import datetime
import pandas as pd
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Common metrics for all routes
metrics.info('app_info', 'Application info', version='1.0.3')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get top 5 repos from dataset
        df = pd.read_csv('github_repo_features_new.csv')
        top_repos = df.sample(5).to_dict('records')

        # Predict stars for each repo
        result = predict_repos(top_repos)
        return render_template('results.html', result=result)

    # For GET request, show single sample prediction
    result = predict_sample()
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100)