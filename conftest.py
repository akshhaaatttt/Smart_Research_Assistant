"""Pytest bootstrap: make the project root importable and apply env defaults.

Placing this at the repository root ensures ``import src.<module>`` works when the
test suite is invoked from any working directory, and that the HuggingFace/TF
environment fix is applied before any model import happens during tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Apply the TF/Keras environment fix before any transformers import in tests.
import src.utils.env  # noqa: E402,F401
