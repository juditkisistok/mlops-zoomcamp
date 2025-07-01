import code.model as model

def test_prepare_features():
    model_service = model.ModelService(None)
    ride = {
        "PULocationID": 130,
        "DOLocationID": 205,
        "trip_distance": 3.66,
    }
    
    actual_features = model_service.prepare_features(ride)
    
    expected_features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }
    
    assert actual_features == expected_features

def test_base64_decode():
    base64_input = "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ=="
    actual_data = model.base64_decode(base64_input)
    expected_data = {
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66,
        },
        "ride_id": 256,
    }

    assert actual_data == expected_data

class MockModel:
    # we want to create a mock model so that we can test the predict function
    # without having to load the model from AWS
    # this way, we can test the predict function in isolation
    
    def __init__(self, value):
        self.value = value
    
    def predict(self, X):
        # return a constant value for each row in the input
        n = len(X)
        return [self.value] * n

def test_predict():
    model_mock = MockModel(10.0)
    model_service = model.ModelService(model_mock)
    features = {
        "PU_DO": "130_205",
        "trip_distance": 3.66,
    }
    
    actual_prediction = model_service.predict(features)
    expected_prediction = 10.0

    assert actual_prediction == expected_prediction

def test_lambda_handler():
    model_mock = MockModel(10.0)
    model_version = "123"
    model_service = model.ModelService(model_mock, model_version)

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                }
            }
        ]
    }
    actual_predictions = model_service.lambda_handler(event, None)
    expected_predictions = {
        "predictions": [
            {
                "model": "ride_duration_prediction_model",
                "version": model_version,
                "prediction": {
                    "ride_duration": 10.0,
                    "ride_id": 256
                }
            }
        ]
    }

    assert actual_predictions == expected_predictions
