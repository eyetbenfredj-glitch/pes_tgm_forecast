"""Entry point for Streamlit Cloud / local runs. Works from ANY working directory."""
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # the app reads data/ and models/ with paths relative to the repo root
runpy.run_path(str(ROOT / "src" / "dashboard" / "app.py"), run_name="__main__")