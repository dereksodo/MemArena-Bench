#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasetgen.cli import main


if __name__ == "__main__":
    main()
