# tests/conftest.py
import sys
from pathlib import Path

# repo_root  = …/svp_cim_project
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")