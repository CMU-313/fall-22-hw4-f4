import json
from flask import jsonify, request
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
        #use entries from the query string here 
        try:
            if (len(request.args) != 0): 
                G1 = request.args['G1']
                G2 = request.args['G2']
                Failures = request.args['Failures']
                Higher = request.args['Higher']
            # entries from json
            else: 
                response = json.loads(request.json)
                print(response)
                G1 = response['G1']
                G2 = response['G2']
                Failures = response['Failures']
                Higher = response['Higher']
        except:
            return 'Missing value', 404

        if (int(G1) < 0 or int(G1) > 20):
            return 'Invalid G1 value', 400
        if (int(G2) < 0 or int(G2) > 20):
            return 'Invalid G2 value', 400
        if (int(Failures) < 1 or int(Failures) > 4):
            return 'Invalid Failures value', 400
        if (not (isinstance(Higher, bool) or (Higher == "False" or Higher == "True"))):
            return 'Invalid Higher value', 400

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

