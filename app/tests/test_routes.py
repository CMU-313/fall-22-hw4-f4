from flask import Flask

from app.handlers.routes import configure_routes


def test_base_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'

# 200, return value is average, above average, or exemplar
def test_predict_success():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G1':5, 'G2':15,'Failures':3, 'Higher':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 200
    assert response.get_data() in set(b'average', b'above average', b'exemplar')

def test_predict_success_edge():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    
    # lower end edge cases
    data={'G1':0, 'G2':0,'Failures':1, 'Higher':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 200
    assert response.get_data() in set(b'average', b'above average', b'exemplar')

    # upper end edge cases
    data={'G1':20, 'G2':20,'Failures':4, 'Higher':True}
    response = client.get(url, query_string=data)

    assert response.status_code == 200
    assert response.get_data() in set(b'average', b'above average', b'exemplar')

# 404, param or params missing
def test_predict_missing_G1():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G2':10, 'Failures':1, 'Higher':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 404

def test_predict_missing_G2():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G1':10, 'Failures':1, 'Higher':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 404

def test_predict_missing_Failures():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G1':10, 'G2':10, 'Higher':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 404

def test_predict_missing_Higher():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G1':10, 'G2':10, 'Failures':1}
    response = client.get(url, query_string=data)
    
    assert response.status_code == 404

def test_predict_missing_params():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'

    # missing multiple params
    data={'G1':10, 'Failures':1}
    response = client.get(url, query_string=data)
    assert response.status_code == 404

    # missing all params
    data={}
    response = client.get(url, query_string=data)
    assert response.status_code == 404

# if some params are missing and others are invalid, status code = 404
def test_predict_missing_and_invalid():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    data={'G1':30, 'G2':10, 'Failures':False}
    response = client.get(url, query_string=data)

    assert response.status_code == 404


# 400, parameters are invalid

def test_predict_invalid_G1():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    
    # G1 too high
    data={'G1':21, 'G2':10, 'Failures':1, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400
    
    # G1 too low
    data={'G1':-1, 'G2':10, 'Failures':1, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

def test_predict_invalid_G2():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'

    # G2 too high
    data={'G1':10, 'G2':21, 'Failures':1, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

    # G2 too low
    data={'G1':10, 'G2':-1, 'Failures':1, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

def test_predict_invalid_Failures():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    
    # Failures too high
    data={'G1':10, 'G2':10, 'Failures':5, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

    # Failures too low
    data={'G1':10, 'G2':10, 'Failures':0, 'Higher':True}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

def test_predict_invalid_Higher():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'
    
    data={'G1':10, 'G2':10, 'Failures':1, 'Higher':'haha'}
    response = client.get(url, query_string=data)
    assert response.status_code == 400

def test_predict_invalid_params():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/predict'

    data={'G1':-5, 'G2':17, 'Failures':10, 'Higher':'oops'}
    response = client.get(url, query_string=data)
    assert response.status_code == 400