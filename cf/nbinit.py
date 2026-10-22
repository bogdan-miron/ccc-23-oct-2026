"""Imported at kernel start by tools/nb.py: common imports + pandas display options + challenge handle `ch`."""
import os, sys, re, time, math, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 60); pd.set_option("display.precision", 4)
ROOT = Path(os.environ.get("CF_ROOT", Path(__file__).resolve().parents[1]))
from cf import io as cio, metrics, models, cv, submit, paths, profile
from cf.cache import disk_cache
ch = paths.Challenge(os.environ["CF_CHALLENGE"]) if os.environ.get("CF_CHALLENGE") else None
def tick(label=""):
    """t = tick(); ...; tick('step', t)  -> prints elapsed seconds"""
    return time.time()
