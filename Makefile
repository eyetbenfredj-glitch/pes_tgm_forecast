install: ; pip install -r requirements.txt
test:    ; pytest tests -q
benchmark: ; python scripts/12_real_benchmark.py
api:     ; uvicorn src.api.main:app --reload --port 8000
dashboard: ; streamlit run src/dashboard/app.py
