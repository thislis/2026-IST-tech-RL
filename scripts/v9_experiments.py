#!/usr/bin/env python3
"""Location-independent v9 CLI; never changes the installed environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blackout_v9.runner import main

if __name__ == "__main__":
    main()
