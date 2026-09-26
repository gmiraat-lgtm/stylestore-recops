# RecOps serving API - production image.
#
# Packages the Flask recommendation service together with the trained model
# artifact, so the container is fully self-contained: no MLflow server or
# registry connection is needed at runtime. The model is baked in at build
# time (CI builds a fresh image after every successful pipeline run), which
# trades hot-reload for immutability - each image IS one model version,
# identified by the MODEL_VERSION build argument. This is the standard
# "immutable artifact" deployment pattern; the hot-reloading dev server in
# serving/app.py remains the local demo path.

FROM python:3.10-slim

WORKDIR /app

# Serving needs only a small subset of the project's requirements.
COPY serving/requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src/recommender.py src/
COPY serving/app_container.py serving/
COPY models/item_cf.pkl models/

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8001
CMD ["python", "serving/app_container.py"]
