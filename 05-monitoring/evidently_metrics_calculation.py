# standard libraries
import datetime
import polars as pl
import logging
import time
import psycopg

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from evidently import ColumnMapping 
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

from joblib import load

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

SEND_TIMEOUT = 10

create_table_statement = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    nr_drifted_columns INTEGER,
    share_missing_values FLOAT
);
"""

reference_data = pl.read_parquet("data/reference.parquet")
with open("models/lin_reg.bin", "rb") as f:
    model = load(f)

# to simulate production usage, we read the data day by day
raw_data = pl.read_parquet("data/green_tripdata_2022-02.parquet")

begin_date = datetime.datetime(2022, 2, 1, 0, 0)

num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]

column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None 
)

report = Report(metrics=[
    ColumnDriftMetric(column_name="prediction"),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
    # access the database with psycopg connect
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres", autocommit=True) as conn:
        res = conn.execute("SELECT 1 from pg_database where datname = 'test'")
        if len(res.fetchall()) == 0:
            # if it returns nothing, it means that the database doesn't exist
            # then we create the database
            conn.execute("CREATE DATABASE test;")
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres dbname=test") as conn:
        conn.execute(create_table_statement)

@task(cache_policy=NO_CACHE)
def calculate_metrics(curr, i):
    current_data = (
        raw_data
        .filter(
            (
                pl.col("lpep_pickup_datetime") >= (begin_date + datetime.timedelta(days=i))
                )
              & (pl.col("lpep_pickup_datetime") < (begin_date + datetime.timedelta(days=i+1))))
              
              ).to_pandas()
    
    current_data.fillna(0, inplace=True)

    current_data["prediction"] = model.predict(current_data[num_features + cat_features])
    
    report.run(reference_data=reference_data.to_pandas(), current_data=current_data, column_mapping=column_mapping)
    
    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]

    nr_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]

    share_missing_values = result["metrics"][2]["result"]["current"]["share_of_missing_values"]

    curr.execute("INSERT INTO dummy_metrics (timestamp, prediction_drift, nr_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)", 
                 (begin_date + datetime.timedelta(days=i), 
                  prediction_drift, nr_drifted_columns, share_missing_values))
    
@flow
def batch_monitoring():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)
    with psycopg.connect("host=localhost port=5432 user=postgres password=postgres dbname=test", autocommit=True) as conn:
        for i in range(27):
            with conn.cursor() as curr:
                calculate_metrics(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send += datetime.timedelta(seconds=SEND_TIMEOUT)
            logging.info(f"Sent metrics at {last_send}")

if __name__ == "__main__":
    batch_monitoring()
