import polars as pl 
import numpy as np
import sys

from sklearn.feature_extraction import DictVectorizer
import os
from dotenv import load_dotenv
import uuid
import mlflow

load_dotenv()

def generate_uuids(df: pl.DataFrame) -> pl.DataFrame:
    # we need to generate a unique ID for the input file
    # normally, we would have a unique ID, but here we just generate a random one
    ride_ids = [str(uuid.uuid4()) for _ in range(len(df))]

    return df.with_columns(pl.Series(name="ride_id", values=ride_ids))

def load_model(run_id: str):
    return mlflow.pyfunc.load_model(f"s3://mlops-bucket-orchestration/992595661440936711/{run_id}/artifacts/model")

def read_dataframe(filename: str) -> pl.DataFrame: 
    """
    Read a dataframe from a parquet file and return a dataframe with the duration of the trip in minutes.
    """
    df = (
        pl.read_parquet(filename)
        .with_columns(
            (pl.col("lpep_dropoff_datetime") - pl.col("lpep_pickup_datetime"))
            .dt.total_seconds()
            .alias("duration")
        )
        .with_columns(
            (pl.col("duration") / 60)
            .alias("duration_minutes")
        )
        .with_columns(
            pl.concat_str(pl.col("PULocationID"), pl.lit("_"), pl.col("DOLocationID")).alias("PU_DO")
        )
        .filter((pl.col("duration_minutes") > 1) & (pl.col("duration_minutes") <= 60))
        .with_columns(pl.col(["PULocationID", "DOLocationID"]).cast(pl.Utf8))
    )

    return generate_uuids(df)

def prepare_dictionaries(df: pl.DataFrame) -> tuple[np.ndarray, DictVectorizer]:
    """
    Prepare the dictionaries for the model.
    """
    return df.select(["PU_DO", "trip_distance"]).to_dicts()

def apply_model(input_file: str, run_id: str, output_file: str) -> None:
    print(f"Reading file {input_file}")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    model = load_model(run_id)
    print(f"Model loaded with run_id {run_id}")
    
    print(f"Predicting {len(dicts)} rows")
    y_pred = model.predict(dicts)

    print(f"Writing results to {output_file}")
    df_result = pl.DataFrame({
    "ride": df.select("ride_id").to_series(),
    "lpep_pickup_datetime": df.select("lpep_pickup_datetime").to_series(),
    "PULocationID": df.select("PULocationID").to_series(),
    "DOLocationID": df.select("DOLocationID").to_series(),
    "actual_duration": df.select("duration_minutes").to_series(),
    "predicted_duration": y_pred,
    "difference": df.select("duration_minutes").to_series() - y_pred,
    "model_version": run_id
})
    df_result.write_parquet(output_file)

def run():
    taxi_type = sys.argv[1] if len(sys.argv) > 1 else "green" 
    year = int(sys.argv[2]) if len(sys.argv) > 3 else 2021
    month = int(sys.argv[3]) if len(sys.argv) > 2 else 1

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{year:04d}-{month:02d}.parquet"

    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

    RUN_ID = os.getenv("RUN_ID", "3d2f54b35a4743b4af6ee34e3415c787")

    apply_model(input_file, RUN_ID, output_file)

if __name__ == "__main__":
    run()
