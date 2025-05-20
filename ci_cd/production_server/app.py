from flask import Flask, render_template, redirect, url_for
from workerA import predict_sample

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    task = predict_sample.delay()
    result = task.get(timeout=10)
    return render_template('results.html', result=result)


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5100,debug=True)
