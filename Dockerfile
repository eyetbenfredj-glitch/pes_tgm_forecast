FROM python:3.11-slim

# libgomp1 is required by LightGBM/XGBoost (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV DATA_MODE=demo PYTHONUNBUFFERED=1
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

# Default: API.  Dashboard: docker run ... streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
