# Build context is the REPO ROOT.
#   docker build -f tensorflow/Dockerfile -t security-ml-tensorflow .
# The source dir is copied to `tensorflow_impl` so it doesn't shadow the
# installed `tensorflow` library on import.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY tensorflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common ./common
COPY tensorflow ./tensorflow_impl
CMD ["python", "-m", "tensorflow_impl.train"]
