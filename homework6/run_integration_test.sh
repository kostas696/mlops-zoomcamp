#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting environment variables..."
export INPUT_FILE_PATTERN="s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"
export AWS_ACCESS_KEY_ID="test"
export AWS_SECRET_ACCESS_KEY="test"
export AWS_DEFAULT_REGION="us-east-1"

echo "Uploading test data and running batch pipeline..."
python tests/integration_test.py

echo "Integration test completed successfully!"
