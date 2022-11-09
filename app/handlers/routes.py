from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
import sklearn

def configure_routes(app):
    df = pd.read_csv("../../data/student-mat.csv")
    df['qual_student'] = np.where(df['G3']>=15, 1, 0)
    include = ['failures','higher','G1','G2','qual_student']
    df.drop(columns=df.columns.difference(include), inplace=True)  # only using selected features

    X = df.drop(['qual_student'], axis = 1)
    y = df['qual_student']
    rfc = RandomForestClassifier(criterion='gini', 
                                n_estimators=200,
                                max_depth=10,
                                min_samples_leaf=6,
                                max_features='auto',
                                oob_score=True,
                                random_state=42,
                                n_jobs=-1,
                                verbose=1)
    rfc.fit(X, y)
    joblib.dump(rfc, 'model.pkl')

    @app.route('/')
    def hello():
        return "try the predict route it is great!"


    @app.route('/predict')
    def predict():
        #use entries from the query string here but could also use json
        failures = request.args.get('failures')
        higher = request.args.get('higher')
        G1 = request.args.get('G1')
        G2 = request.args.get('G2')
        data = [[failures], [higher], [G1], [G2]]
        query_df = pd.DataFrame({
            'failures': pd.Series(failures),
            'higher': pd.Series(higher),
            'G1': pd.Series(G1),
            'G2': pd.Series(G2),
        })
        query = pd.get_dummies(query_df)
        prediction = rfc.predict(query)
        return jsonify(np.ndarray.item(prediction))