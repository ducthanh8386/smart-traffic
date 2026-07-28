from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT_DIR / ".runtime"


def configure_runtime() -> None:
    """Configure writable local cache directories for third-party libraries."""
    import sys

    venv_site_packages = ROOT_DIR / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    ultralytics_dir = RUNTIME_DIR / "ultralytics"
    matplotlib_dir = RUNTIME_DIR / "matplotlib"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

