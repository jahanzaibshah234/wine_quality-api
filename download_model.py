import boto3
import os

BUCKET_NAME = "jahanzaib-mlops-2026"
MODEL_KEY = "models/wine_quality.pkl"
MODEL_PATH = "wine_quality.pkl"


def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists. Skipping download.")
        return

    print("Downloading model from S3...")

    s3 = boto3.client("s3")

    s3.download_file(
        BUCKET_NAME,
        MODEL_KEY,
        MODEL_PATH
    )

    print("Model downloaded successfully.")


if __name__ == "__main__":
    download_model()
