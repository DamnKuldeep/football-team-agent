"""Entry point for Streamlit (Community Cloud or `streamlit run streamlit_app.py`).

Puts src/ on the path and runs the app script on every rerun, so no package
install is needed.
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "fta" / "ui_app.py"), run_name="__main__")
