"""migratory-birds level 6. Top 50 BOPs by Arrivals on day 791 after the "WeatherChanger".
Finding: days 761..790 are an exact time mirror of days 731..760 (Arrivals and Occupancy cells identical, wind negated): day 760+k = day 761-k.
The world runs backwards, so day 791 = day 730. Prediction = top 50 of day 730 (ties by lower BOP id). solve() verifies the mirror on all 30 days first;
if it does not hold it falls back to persistence (top 50 of day 790)."""
from pathlib import Path
import numpy as np, pandas as pd

HEADER = "Day,Top 50 Arrivals BOPs"
N0 = 760          # last day of the old regime; new regime = days 761..790
NEW = 30

def load(level_dir, path):
    def rd(p):
        df = pd.read_csv(p, na_values=["missing"], dtype={"Day": "int16", "BOP": "int16"}, skipinitialspace=True); df.columns = ["day", "bop", "arr", "occ", "wx", "wy", "ins"]; return df
    full = pd.concat([rd(Path(level_dir) / "all_data_from_level_5.in"), rd(path)], ignore_index=True)
    piv = lambda c: full.pivot(index="day", columns="bop", values=c).values.astype("float32")
    return piv("arr"), piv("occ"), full.groupby("day")[["wx", "wy"]].first().values.astype("float32")

def mirror_holds(A, O, W):
    return all(np.array_equal(A[N0 + k], A[N0 - 1 - k]) and np.array_equal(O[N0 + k], O[N0 - 1 - k]) and np.allclose(W[N0 + k], -W[N0 - 1 - k]) for k in range(NEW))

def rank50(row):
    return np.lexsort((np.arange(len(row)), -row))[:50] + 1      # by Arrivals desc, ties: lower BOP id first

def predict(A, O, W, mirror=True):
    day = N0 + NEW + 1                                           # 791
    if mirror and mirror_holds(A, O, W): src = N0 - NEW         # day 730 (1-based) -> index 729
    else: src = N0 + NEW                                         # day 790
    return day, src, rank50(A[src - 1])

def solve(path, ctx) -> str:
    A, O, W = load(ctx.level.dir, path); day, src, top = predict(A, O, W)
    return "\n".join([HEADER, f"{day},{' '.join(map(str, top))}"])
