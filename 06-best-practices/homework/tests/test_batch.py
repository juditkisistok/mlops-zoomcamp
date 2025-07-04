import batch
from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    # in prepare_data, we want to filter out rows where the duration is less than 1 minute or greater than 60 minutes
    # so here, 2 out of 4 rows should be filtered out
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    df_prepared = batch.prepare_data(df, columns)

    assert df_prepared.shape[0] == 2