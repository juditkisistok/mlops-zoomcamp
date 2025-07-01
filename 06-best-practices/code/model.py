import json
import boto3
import base64

import mlflow

#kinesis_client = boto3.client('kinesis')

def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode('utf-8')
    return json.loads(decoded_data)

class ModelService():

    def __init__(self, model, model_version = None, callbacks = None):
        # callbacks is a list of functions that will be called after each prediction is made
        # for example, one callback could be to put into a kinesis stream
        self.model = model
        self.model_version = model_version
        self.callbacks = callbacks or []

    def lambda_handler(self, event, context):
    
        predictions_events = []
        
        for record in event['Records']:
            encoded_data = record['kinesis']['data']
            ride_event = base64_decode(encoded_data)

            # print(ride_event)
            ride = ride_event['ride']
            ride_id = ride_event['ride_id']
        
            features = self.prepare_features(ride)
            prediction = self.predict(features)
        
            prediction_event = {
                'model': 'ride_duration_prediction_model',
                'version': self.model_version,
                'prediction': {
                    'ride_duration': prediction,
                    'ride_id': ride_id   
                }
            }

            for callback in self.callbacks:
                callback(prediction_event)
            
            predictions_events.append(prediction_event)


        return {
            'predictions': predictions_events
        }
    
    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features


    def predict(self,features):
        pred = self.model.predict(features)
        return float(pred[0])

def load_model(run_id):
    logged_model = f's3://mlflow-models-alexey/1/{run_id}/artifacts/model'
    model = mlflow.pyfunc.load_model(logged_model)
    return model

class KinesisCallback:
    def __init__(self, kinesis_client, predictions_stream_name):
        self.kinesis_client = kinesis_client
        self.predictions_stream_name = predictions_stream_name
    
    def put_record(self, prediction_event):
        ride_id = prediction_event['prediction']['ride_id']
        
        self.kinesis_client.put_record(
            StreamName=self.predictions_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=str(ride_id)
        )

def init(run_id, test_run, predictions_stream_name):
    model = load_model(run_id)

    callbacks = []

    if not test_run:
        kinesis_client = boto3.client('kinesis')
        kinesis_callback = KinesisCallback(kinesis_client, predictions_stream_name)
        callbacks.append(kinesis_callback.put_record)

    return ModelService(model, callbacks)