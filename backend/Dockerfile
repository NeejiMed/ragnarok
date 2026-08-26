#  Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY backend/ backend/

RUN pip install --no-cache-dir .

#  Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy BOTH site-packages AND bin (executables like uvicorn live in bin)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY --from=builder /app /app

RUN mkdir -p data/uploads

RUN useradd -m -u 1000 ragnarok && chown -R ragnarok:ragnarok /app
USER ragnarok

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
