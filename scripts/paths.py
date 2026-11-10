"""Shared configurable paths; safe for CPU-only preparation tools."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("T3_WORK_DIR", str(Path.home() / ".local/share/qwen-flash-trellis")))
FP8_SRC = Path(os.environ.get("T3_FP8_SOURCE", str(ROOT / "source")))
LIVE = Path(os.environ.get("T3_BASE_MODEL", str(ROOT / "base")))
