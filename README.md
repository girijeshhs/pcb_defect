# Work Progress v1

# Terminal 1 — FastAPI backend (port 8000)
cd "/Users/girijesh/Desktop/Requisit/Repos/pcb models"
python3 -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Streamlit frontend
cd "/Users/girijesh/Desktop/Requisit/Repos/pcb models"
python3 -m streamlit run app.py
