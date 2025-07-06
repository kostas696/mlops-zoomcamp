import pandas as pd
from datetime import datetime
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),          
        (1, 1, dt(1, 2), dt(1, 10)),                
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),       
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),          
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_processed = prepare_data(df, categorical)

    # We expect only the first 2 rows to remain
    assert len(df_processed) == 2

    # Check transformed types
    assert df_processed['PULocationID'].dtype == 'object'
    assert df_processed['DOLocationID'].dtype == 'object'

    # Optional: convert to list of dicts for easier visibility
    expected_dicts = [
        {'PULocationID': '-1', 'DOLocationID': '-1'},
        {'PULocationID': '1',  'DOLocationID': '1'},
    ]
    actual_dicts = df_processed[categorical].to_dict(orient='records')

    assert actual_dicts == expected_dicts
