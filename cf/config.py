"""Machine-dependent knobs. Override with env vars."""
import os
NJOBS = int(os.environ.get("CF_JOBS", min(os.cpu_count() or 2, 8)))   # threads for GBMs / forests
NB_TIMEOUT = int(os.environ.get("CF_NB_TIMEOUT", 165))                  # seconds per notebook cell (keep < the shell's own call cap)
