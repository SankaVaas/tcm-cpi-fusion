#!/usr/bin/env python
"""Step 1: Download all public datasets (except TCMSP — see docs/DATA_SOURCES.md)."""
import sys
sys.path.insert(0, ".")
from src.data.download import download_all

if __name__ == "__main__":
    download_all()
