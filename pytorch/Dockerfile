# Build context is the REPO ROOT so we can copy common/ alongside pytorch/.
#   docker build -f pytorch/Dockerfile -t security-ml-pytorch .
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# CPU-only torch wheel (far smaller than the default CUDA build).
COPY pytorch/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
 && pip install --no-cache-dir numpy==1.26.4
COPY common ./common
COPY pytorch ./pytorch
CMD ["python", "-m", "pytorch.train"]
