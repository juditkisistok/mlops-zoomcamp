#import predict
import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}

#features = predict.prepare_features(ride)
#pred = predict.predict(features)
#print(pred)

response = requests.post(url="http://127.0.0.1:9696/predict", json=ride)
print(response.json())