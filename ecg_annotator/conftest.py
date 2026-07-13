import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Allow Qt widgets to be created in tests without a real display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
