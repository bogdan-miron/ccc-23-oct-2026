"""migratory-birds level 1. Sort BOPs most to least popular: temperature high first, then humidity low first, then BOP id low first.
Output: one line, BOP ids separated by single spaces. solve() = raw data (variant A). solve_fixF() = values above 45 read as degree F (variant B)."""
from pathlib import Path
import pandas as pd
from cf import io as cio

F_GAP = 45  # file e: no values in 46..54; values >= 55 are degree F (max C is 45, max F 113)

def load(path):
    df = cio.read_table(Path(path))
    df.columns = ["bop", "temp", "hum"]
    return df

def to_celsius(temp):
    """Level 2 PDF confirms: some values in the Celsius column are degree F. Values above F_GAP (45) are F. Returns a float Series."""
    t = pd.Series(temp, dtype=float)
    return t.where(t <= F_GAP, (t - 32) * 5 / 9)

def rank(df, fix_f=False):
    d = df.copy()
    if fix_f:
        m = d.temp > F_GAP
        d.loc[m, "temp"] = (d.loc[m, "temp"] - 32) * 5 / 9
    d = d.sort_values(["temp", "hum", "bop"], ascending=[False, True, True], kind="mergesort")
    return " ".join(str(int(b)) for b in d.bop)

def solve(path, ctx) -> str:
    return rank(load(path))

def solve_fixF(path, ctx) -> str:
    return rank(load(path), fix_f=True)
