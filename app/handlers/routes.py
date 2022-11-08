import this
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import os

def configure_routes(app):

    this_dir = os.path.dirname(__file__)
    model_path = os.path.join(this_dir, "model.pkl")
    clf = joblib.load(model_path)

    @app.route('/')
    def hello():
        return "try the predict route it is great!"


    @app.route('/predict')
    def predict():
        #use entries from the query string here but could also use json
        G1 = request.args.get('G1')
        G2 = request.args.get('G2')
        Failures = request.args.get('Failures')
        Higher = request.args.get('Higher')
        data = [[G1], [G2], [Failures], [Higher]]
        query_df = pd.DataFrame({
            'G1': pd.Series(G1),
            'G2': pd.Series(G2),
            'Failures': pd.Series(Failures),
            'Higher': pd.Series(Higher)
        })
        query = pd.get_dummies(query_df)
        prediction = clf.predict(query)
        prediction = np.ndarray.item(prediction)
        category = ''
        if (prediction == 1):
            category = 'average'
        elif (prediction == 2):
            category = 'above average'
        elif (prediction == 3):
            category = 'exemplar'
        else:
            return 'Invalid prediction calculated', 400
        return jsonify(category)