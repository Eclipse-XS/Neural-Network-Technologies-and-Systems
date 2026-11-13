"""Paths are anchored to this editable installation, never the working directory."""
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[2]
