import argparse
import pickle

from pathlib import Path

import mlflow
import polars as pl
import xgboost as xgb
import numpy as np

from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from mlflow.models.signature import infer_signature

from prefect import task, flow

"""
Deployment commands:
prefect init - create the prefect.yaml file
prefect server start - start the server
prefect deploy 03-orchestration/duration-prediction.py:run -n taxi-flow -p "mlops-pool" -create the deployment
prefect worker start -p "mlops-pool"  - start the worker
"""

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_dataframe(year: int, month: int, color: str = "green") -> pl.DataFrame: 
    """
    Read a dataframe from a parquet file and return a dataframe with the duration of the trip in minutes.
    """

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
    
    df = (
        pl.read_parquet(url)
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

    return df

@task(log_prints=True)
def create_X(df: pl.DataFrame, dv: DictVectorizer = None) -> tuple[np.ndarray, DictVectorizer]:
    """
    Create a feature matrix from a dataframe and a dictionary vectorizer.
    """
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df.select(categorical + numerical).to_dicts()

    if dv is None:
        dv = DictVectorizer(sparse = True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

@task(log_prints=True)
def train_model(X_train: xgb.DMatrix, y_train: np.ndarray, 
                X_val: xgb.DMatrix, y_val: np.ndarray, dv: DictVectorizer) -> str:
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        params = {
            "reg_lambda": 0.10479357768633932,
            "seed": 42,
            "max_depth": 67,
            "min_child_weight": 4.726134327631799,
            "learning_rate": 0.2254998660071301,
            "objective": "reg:squarederror",
            "reg_alpha": 0.012770048180595262,
        }

        mlflow.log_params(params)

        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=30,
                evals=[(valid, "validation")],
                early_stopping_rounds=50,
            )
        
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open(models_folder / "preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(models_folder / "preprocessor.b", artifact_path="preprocessor")
        
        signature = infer_signature(X_val, y_pred)
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow", signature=signature, model_format="json")

        # return the run id from mlflow
        return run.info.run_id
        
@flow(log_prints=True, name="NYC-Taxi-Duration-Prediction")
def run(year: int, month: int) -> str:
    """
    Run the NYC Taxi Duration Prediction model.
    """
    df_train = read_dataframe(year, month)

    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1

    df_val = read_dataframe(next_year, next_month)

    print(f"len(df_train): {len(df_train)}, len(df_val): {len(df_val)}")

    X_train, dv = create_X(df_train, None)
    X_val, _ = create_X(df_val, dv)

    target = "duration_minutes"
    y_train, y_val = df_train[target], df_val[target]

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    return run_id

if __name__ == "__main__":
    # use argparse to get the year and month from the command line
    parser = argparse.ArgumentParser(description="Train a model to predict the duration of a taxi trip")
    
    parser.add_argument("--year", type=int, default=2021, help="The year of the data to use")
    parser.add_argument("--month", type=int, default=1, help="The month of the data to use")
    args = parser.parse_args()
    
    run_id = run(year = args.year, month = args.month)
    # save run id to a file
    with open("run_id.txt", "w") as f:
        f.write(run_id)
