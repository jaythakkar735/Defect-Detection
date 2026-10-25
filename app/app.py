"""
Legacy Entrypoint Forwarder.
Routes to the main industrial application at the project root: app.py
"""
import runpy
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

root_app = root_dir / "app.py"
runpy.run_path(str(root_app), run_name="__main__")
