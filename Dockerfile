FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy dependency file first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

# Copy full package
COPY . .

# Expose port
EXPOSE 7860

# Liveness probe — validator checks this
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
