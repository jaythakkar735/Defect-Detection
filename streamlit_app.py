"""
Streamlit Community Cloud Default Entrypoint.
Auto-detected by Streamlit Community Cloud during 1-click deployment.
Routes to the main industrial application at app.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import and execute app directly
import app
