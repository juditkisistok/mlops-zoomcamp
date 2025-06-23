import datetime
import polars as pl
import logging
import time
import psycopg

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    DataQualityStabilityMetric,
)

from joblib import load

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)


@task(cache_policy=NO_CACHE)
def prep_db():
    create_table_statement = """
    DROP TABLE IF EXISTS hw_metrics;
    CREATE TABLE hw_metrics (
        timestamp TIMESTAMP,
        prediction_drift FLOAT,
        nr_drifted_columns INTEGER,
        share_missing_values FLOAT,
        quantile_metric FLOAT,
        not_stable_prediction INTEGER
    );
    """

    # access the database with psycopg connect
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=postgres", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 from pg_database where datname = 'test'")
        if len(res.fetchall()) == 0:
            # if it returns nothing, it means that the database doesn't exist
            # then we create the database
            conn.execute("CREATE DATABASE test;")
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=postgres dbname=test"
    ) as conn:
        conn.execute(create_table_statement)


@task(cache_policy=NO_CACHE)
def prep_report(num_features, cat_features):
    column_mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=num_features,
        categorical_features=cat_features,
        target=None,
    )

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="prediction"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataQualityStabilityMetric(),
            ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
        ]
    )

    return report, column_mapping


@task(cache_policy=NO_CACHE)
def prep_data(raw_data, model, num_features, cat_features, i, begin_date):
    current_data = (
        raw_data.filter(
            (
                pl.col("lpep_pickup_datetime")
                >= (begin_date + datetime.timedelta(days=i))
            )
            & (
                pl.col("lpep_pickup_datetime")
                < (begin_date + datetime.timedelta(days=i + 1))
            )
        )
    ).to_pandas()

    current_data.fillna(0, inplace=True)

    current_data["prediction"] = model.predict(
        current_data[num_features + cat_features]
    )

    return current_data


@task(cache_policy=NO_CACHE)
def calculate_metrics(
    curr, raw_data, i, num_features, cat_features, reference_data, model, begin_date
):
    current_data = prep_data(raw_data, model, num_features, cat_features, i, begin_date)
    report, column_mapping = prep_report(num_features, cat_features)

    report.run(
        reference_data=reference_data.to_pandas(),
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]

    nr_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]

    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]

    not_stable_prediction = result["metrics"][3]["result"][
        "number_not_stable_prediction"
    ]

    quantile_metric = result["metrics"][4]["result"]["current"]["value"]

    curr.execute(
        "INSERT INTO hw_metrics (timestamp, prediction_drift, nr_drifted_columns, share_missing_values, quantile_metric, not_stable_prediction) VALUES (%s, %s, %s, %s, %s, %s)",
        (
            begin_date + datetime.timedelta(days=i),
            prediction_drift,
            nr_drifted_columns,
            share_missing_values,
            quantile_metric,
            not_stable_prediction,
        ),
    )


@flow
def batch_monitoring(
    raw_data,
    reference_data,
    model,
    num_features,
    cat_features,
    begin_date,
    send_timeout=10,
    days=30,
):
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=send_timeout)
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=postgres dbname=test",
        autocommit=True,
    ) as conn:
        for i in range(days):
            with conn.cursor() as curr:
                calculate_metrics(
                    curr,
                    raw_data,
                    i,
                    num_features,
                    cat_features,
                    reference_data,
                    model,
                    begin_date,
                )

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < send_timeout:
                time.sleep(send_timeout - seconds_elapsed)
            while last_send < new_send:
                last_send += datetime.timedelta(seconds=send_timeout)
            logging.info(f"Sent metrics at {last_send}")


if __name__ == "__main__":
    reference_data = pl.read_parquet("data/reference.parquet")
    with open("models/lin_reg.bin", "rb") as f:
        model = load(f)

    raw_data = pl.read_parquet("data/green_tripdata_2024-03.parquet")

    begin_date = datetime.datetime(2024, 3, 1, 0, 0)

    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    batch_monitoring(
        raw_data, reference_data, model, num_features, cat_features, begin_date
    )
