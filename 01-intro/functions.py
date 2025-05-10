import polars as pl
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

def read_dataframe(data, 
                   categorical_features = ["PULocationID", "DOLocationID", "PU_DO"], 
                   pu_col = "lpep_pickup_datetime", do_col = "lpep_dropoff_datetime"): 

    df = (
        data
        # trip type 2 means "dispatch" (not street hail)
        # .filter(pl.col("trip_type") == 2)
        # in pandas, we would need to do pd.to_datetime(df["lpep_pickup_datetime"])
        # but polars parses the datetime correctly
        .with_columns(
            (pl.col(do_col) - pl.col(pu_col))
            # first convert to seconds, then convert to minutes
            # this is a bit more precise than using .dt.total_minutes()
            # we preserve the fractional part
            .dt.total_seconds()
            .alias("duration")
        )
        .with_columns(
            (pl.col("duration") / 60)
            .alias("duration_minutes")
        )
        # add an interaction feature
        .with_columns(
            pl.concat_str(pl.col("PULocationID"), pl.lit("_"), pl.col("DOLocationID")).alias("PU_DO")
        )
    )

    df_filtered = (
        df
        .filter((pl.col("duration_minutes") > 1) & (pl.col("duration_minutes") <= 60))
        # we want to one-hot encode the categorical features
        # but first we need to convert them to strings
        .with_columns(pl.col(categorical_features).cast(pl.Utf8))
    )

    return df, df_filtered, categorical_features

def one_hot_encoding(df_train, df_val, categorical_features, numerical_features):
    # train a DictVectorizer - turns a dictionary into a vector
    # for this to work, we need to convert the dataframe into dictionaries
    dict_vectorizer = DictVectorizer()

    # fit the DictVectorizer on the training data
    X_train = dict_vectorizer.fit_transform(df_train.select(categorical_features + numerical_features).to_dicts())
    # we can look at the features by calling the get_feature_names_out method
    dict_vectorizer.get_feature_names_out()

    X_val = dict_vectorizer.transform(df_val.select(categorical_features + numerical_features).to_dicts())

    return X_train, X_val, dict_vectorizer

def train_model(model, X_train, y_train, X_val, y_val):
    # train a model
    lr = model
    lr.fit(X_train, y_train)

    # make predictions
    y_pred = lr.predict(X_val)
    # calculate the root mean squared error to assess the model's performance
    rmse = root_mean_squared_error(y_val, y_pred)

    sns.distplot(y_pred, label="prediction")
    sns.distplot(y_train, label="actual")
    plt.title(f"Model: {model.__class__.__name__}, RMSE: {rmse}")
    plt.legend()

    return lr