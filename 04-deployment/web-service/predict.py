# Pipfile: it contains the dependencies for the project
# Pipfile.lock: it contains the exact versions of the dependencies

import pickle
import os
from flask import Flask, request, jsonify

print(os.getcwd())

with open("lin_reg.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

app = Flask("duration-prediction")

# decorator turns the function into a HTTP endpoint
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # wrapper to take an HTTP request and return a prediction
    
    # body of the request will contain info about the ride
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": pred
    }

    return jsonify(result)

if __name__ == "__main__":
    # in production, we want to use gunicorn instead
    # gunicorn --bind 0.0.0.0:9696 predict:app
    app.run(debug=True, host="0.0.0.0", port=9696)
