
import numpy as np
import os
from flask import Flask

def detect_anomoly():
    from firebase import firebase

    firebase = firebase.FirebaseApplication('https://hackmit-df9ea.firebaseio.com', None)
    result = firebase.get('/Test/Double', None)
    items = result.items()
    result = np.array(list(result.values())[-60:]).reshape(-1, 1)

    # normalize result
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    result2 = scaler.fit_transform(result)
    from sklearn.covariance import EllipticEnvelope
    clf = EllipticEnvelope(contamination=.3)
    y_Pred = clf.fit_predict(result2)

    an_outlier = len(list(filter(lambda x: (x< 0), y_Pred))) > 0
    if clf.precision_ > 100 and an_outlier:
        return True
    return False

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/detect-anamoly')
    def hello():
        return str(detect_anomoly())

    return app