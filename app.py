import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd




app = Flask(__name__)
model = pickle.load(open('random_forest_pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    
    for k, v in data.items():
        if not isinstance(v, list):
            data[k] = [v]
    sample_df = pd.DataFrame(data)
    output = model.predict(sample_df)
   
    return jsonify(output[0])

if __name__ == "__main__":
    # The line `app.run(debug=True, host="0.0.0")` in the Python Flask code is starting the Flask
    # development server. Here's what each argument does:
    app.run(debug=True,host="0.0.0.0",port=5000)
    
