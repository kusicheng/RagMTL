FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/data/models TMPDIR=/data/tmp OMP_NUM_THREADS=4
COPY pyproject.toml README.md ./
COPY magra_rag ./magra_rag
COPY magra_benchmark ./magra_benchmark
RUN pip install --no-cache-dir . && mkdir -p /data/tmp
ENTRYPOINT ["magra", "--data-root", "/data"]
CMD ["doctor"]
