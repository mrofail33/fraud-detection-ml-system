FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir . fastapi uvicorn

EXPOSE 8000

CMD ["uvicorn", "fraud_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]
