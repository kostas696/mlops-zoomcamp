import argparse
import pickle
import pandas as pd

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    # Read the data for the specified year and month
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet', categorical)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    # Create the ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Create the results dataframe
    df_result = pd.DataFrame({'ride_id': df['ride_id'], 'predicted_duration': y_pred})

    # Save the results to a parquet file
    output_file = f'predictions_{year:04d}_{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    # Print the mean predicted duration
    mean_pred_duration = y_pred.mean()
    print(f'Mean predicted duration for {year:04d}-{month:02d}: {mean_pred_duration}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year of the data to process')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to process')
    args = parser.parse_args()
    
    main(args.year, args.month)
