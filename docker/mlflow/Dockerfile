FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary \
    boto3

ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

EXPOSE 5001

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@mlflow-db:5432/mlflow", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--serve-artifacts"]
