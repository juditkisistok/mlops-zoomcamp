import pickle
import pandas as pd
import os
import sys

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(df, categorical, dv, model):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"The mean of the predicted duration is {y_pred.mean():.3f}.")
    print(f"The standard deviation of the predicted duration is {y_pred.std():.3f}.")

    return y_pred

def prepare_output(df, y_pred, year, month, output_file):
    # Preparing the output
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    def bytes_to_mb(size_in_bytes):
        return size_in_bytes / (1024 * 1024)

    print(f"The size of the output file is {bytes_to_mb(os.path.getsize(output_file)):.2f} MB.")

    return output_file

def run():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    month = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    os.makedirs('output', exist_ok=True)
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet', categorical)
    y_pred = predict(df, categorical, dv, model)
    output_file = prepare_output(df, y_pred, year, month, output_file)


if __name__ == "__main__":
    run()





