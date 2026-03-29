FROM openenv-base:latest

WORKDIR /app

# Copy dependency file first for layer caching
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full package
COPY . .

# Expose port
EXPOSE 8000

# Liveness probe — validator checks this
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
