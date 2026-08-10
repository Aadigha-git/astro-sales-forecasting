FROM astrocrpublic.azurecr.io/runtime:3.0-5

# System deps for LightGBM / compiled extensions (rebuild only when this layer changes)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER astro

ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV AWS_DEFAULT_REGION=us-east-1
ENV PYTHONPATH=/usr/local/airflow/include:${PYTHONPATH}
