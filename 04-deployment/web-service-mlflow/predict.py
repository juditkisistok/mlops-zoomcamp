# Pipfile: it contains the dependencies for the project
# Pipfile.lock: it contains the exact versions of the dependencies

import os
import mlflow
from flask import Flask, request, jsonify

# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# logged_model = f'runs:/{RUN_ID}/model'
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# we can point to the model in S3 - this removes the dependency on the tracking server
RUN_ID = os.getenv("RUN_ID")
logged_model = f"s3://mlops-bucket-orchestration/992595661440936711/{RUN_ID}/artifacts/model"

# get the DictVectorizer as an artifact
# client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
# path = client.download_artifacts(RUN_ID, path="model/dict_vectorizer.bin")
# print("Downloaded DictVectorizer from MLflow to", path)

# with open(path, "rb") as f_in:
#     dv = pickle.load(f_in)

# Load model as a PyFuncModel
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features

def predict(features):
    preds = model.predict(features)
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
        "duration": pred,
        "model_version": RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    # in production, we want to use gunicorn instead
    # gunicorn --bind 0.0.0.0:9696 predict:app
    app.run(debug=True, host="0.0.0.0", port=9696)
