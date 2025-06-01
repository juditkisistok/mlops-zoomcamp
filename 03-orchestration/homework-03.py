import prefect
from prefect import flow, task
import polars as pl
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from prefect_aws import S3Bucket
import pickle

@task(log_prints=True)
def read_dataframe(year: int = 2023, month: int = 3, color: str = "yellow") -> pl.DataFrame:
    """
    Read and process a dataframe from a URL.
    """
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month:02d}.parquet"
    df = pl.read_parquet(url)

    print(f"Number of rows before filtering: {len(df)}")

    df = (
        df
        .with_columns(
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
            .dt.total_seconds()
            .alias("duration")
        )
        .with_columns(
            (pl.col("duration") / 60)
            .alias("duration_minutes")
        )
        .filter((pl.col("duration_minutes") >= 1) & (pl.col("duration_minutes") <= 60))
        .with_columns(pl.col(["PULocationID", "DOLocationID"]).cast(pl.Utf8))
    )

    print(f"Number of rows after filtering: {len(df)}")

    return df

@task(log_prints=True)
def train_and_save_model(df: pl.DataFrame) -> tuple[LinearRegression, DictVectorizer]:
    """
    Train a linear regression model and log it to MLflow.
    """
    with mlflow.start_run():
        dicts = df.select(["PULocationID", "DOLocationID"]).to_dicts()
        dv = DictVectorizer(sparse = True)
        
        X = dv.fit_transform(dicts)
        y = df.select("duration_minutes").to_numpy()

        lr = LinearRegression()
        lr.fit(X, y)

        print(f"Model intercept: {lr.intercept_}")

        mlflow.sklearn.log_model(lr, artifact_path="model")
        mlflow.log_metric("intercept", lr.intercept_)

    return lr, dv

@task(log_prints=True)
def upload_model_to_s3(lr: LinearRegression, dv: DictVectorizer) -> None:
    """
    Upload the model and the DictVectorizer to S3.
    """

    s3_bucket = S3Bucket.load("s3-bucket-mlops")

    for model, name in zip([dv, lr], ["preprocessor.b", "model.pkl"]):
        with open(f"models/{name}", "wb") as f_out:
            pickle.dump(model, f_out)
        s3_bucket.upload_from_path(
            from_path=f"models/{name}",
            to_path=f"models/{name}"
        )

@flow(log_prints=True, name="homework-03")
def run():
    """
    Orchestrate the flow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlops-zoomcamp-homework-03")
    
    print(f"Orchestrator used: Prefect")
    print(f"Prefect version: {prefect.__version__}")
    
    df_train = read_dataframe()
    lr, dv = train_and_save_model(df_train)

    upload_model_to_s3(lr, dv)

if __name__ == "__main__":
    run()

