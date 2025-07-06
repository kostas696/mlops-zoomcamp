import pandas as pd
import os
import sys
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def save_data(df, path):
    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
        }
    }
    df.to_parquet(
        path,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def read_data(path):
    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
        }
    }
    return pd.read_parquet(path, storage_options=options)

# Step 1: Create and save input data
data = [
    (None, None, dt(1, 1), dt(1, 10)),          
    (1, 1, dt(1, 2), dt(1, 10)),                
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),       
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),           
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
df_input = pd.DataFrame(data, columns=columns)

input_path = "s3://nyc-duration/in/2023-01.parquet"
output_path = "s3://nyc-duration/out/2023-01.parquet"

save_data(df_input, input_path)

# Step 2: Run batch.py on the test input
exit_code = os.system(f"python batch.py 2023 1")
assert exit_code == 0, "batch.py failed"

# Step 3: Read output and compute sum of predicted durations
df_result = read_data(output_path)

print("Output DataFrame:")
print(df_result)

total_duration = df_result['predicted_duration'].sum()
print(f"\nSum of predicted durations: {total_duration:.2f}")