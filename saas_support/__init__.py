"""A custom tau2-bench dual-control domain for B2B SaaS support.

Importing this package sets `TAU2_DATA_DIR` before anything under `tau2` is
imported. tau2 computes its `DATA_DIR` at import time relative to a source
checkout; because it is installed here from git, that directory does not
exist, so it is pointed at the small vendored copy in `tau2_data/`.

Scripts must import `saas_support` (or something from it) *before* importing
`tau2`.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("TAU2_DATA_DIR", str(PROJECT_ROOT / "tau2_data"))
