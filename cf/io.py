"""Robust readers for contest files. Handles: 'missing' tokens, numbers spelled as words ('seventeen'), units in headers,
space-separated list columns (paths), big CSVs. Always report what was coerced: silent cleaning hides traps."""
import re
import numpy as np, pandas as pd

MISSING = ["missing", "Missing", "MISSING", "", "NA", "N/A", "nan", "NaN", "null", "None", "?", "-"]

_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_DE = {"null": 0, "eins": 1, "ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fuenf": 5, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
       "elf": 11, "zwoelf": 12, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fuenfzehn": 15, "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19, "zwanzig": 20}

def word2num(s):
    """'seventeen'->17, 'twenty-one'->21, 'minus five'->-5, 'one hundred five'->105; German 0-20. Returns None if not parseable."""
    if not isinstance(s, str): return None
    t = s.strip().lower().replace("-", " ").replace(",", " ")
    if t in _DE: return _DE[t]
    neg = False; toks = [w for w in t.split() if w != "and"]
    if toks and toks[0] in ("minus", "negative"): neg, toks = True, toks[1:]
    if not toks: return None
    total = cur = 0
    for w in toks:
        if w in _ONES: cur += _ONES[w]
        elif w in _TENS: cur += _TENS[w]
        elif w == "hundred": cur = max(cur, 1) * 100
        elif w == "thousand": total += max(cur, 1) * 1000; cur = 0
        else: return None
    v = total + cur
    return -v if neg else v

def to_number(series, mapping=None, report=True):
    """Coerce a column to numeric. Returns (numeric_series, changes_df). changes_df lists every non-numeric token converted/failed."""
    mapping = mapping or {}
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    bad = s.notna() & num.isna() & ~s.astype(str).isin(MISSING)
    ch = []
    for idx in s.index[bad]:
        raw = s[idx]; v = mapping.get(raw, word2num(raw))
        if v is not None: num[idx] = v
        ch.append((idx, raw, v))
    changes = pd.DataFrame(ch, columns=["row", "raw", "parsed"])
    if report and len(changes): print(f"[to_number] {series.name}: {len(changes)} non-numeric tokens -> {changes['parsed'].notna().sum()} parsed, {changes['parsed'].isna().sum()} unparsed; e.g. {changes.head(5).values.tolist()}")
    return num, changes

def read_table(path, numeric=True, **kw):
    """Read a contest CSV (header, comma). 'missing' -> NaN. Object columns that are mostly numeric are coerced (with report).
    Returns DataFrame; df.attrs['raw_columns'] keeps original header strings (needed to write outputs); df.attrs['dirty'] lists coerced tokens."""
    df = pd.read_csv(path, na_values=MISSING, keep_default_na=False, skipinitialspace=True, **kw)
    df.attrs["raw_columns"] = list(df.columns); df.attrs["dirty"] = {}
    if numeric:
        for c in df.columns[df.dtypes == object]:
            nn = df[c].notna().sum()
            if nn == 0: continue
            parsed = pd.to_numeric(df[c], errors="coerce").notna().sum()
            wordy = df[c].dropna().map(lambda x: word2num(x) is not None).sum()
            if (parsed + wordy) / nn > 0.9 and parsed > 0:
                df[c], chg = to_number(df[c], report=False)
                if len(chg): df.attrs["dirty"][c] = chg; print(f"[read_table] {path.name if hasattr(path,'name') else path}: column '{c}' coerced to numeric, {len(chg)} tokens fixed: {chg.head(4).values.tolist()}")
    return df

def snake(col):
    """'Temperature [°C]' -> 'temperature_c'; 'Bird Love Score [<3]' -> 'bird_love_score_3' ; use rename map, keep raw names in attrs."""
    c = re.sub(r"\[([^\]]*)\]", r" \1", str(col)); c = c.replace("°", "").replace("²", "2").replace("%", "pct").replace("<", "lt").replace(">", "gt")
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c)).strip("_").lower()

def snake_columns(df):
    raw = df.attrs.get("raw_columns", list(df.columns)); out = df.rename(columns={c: snake(c) for c in df.columns}); out.attrs.update(df.attrs); out.attrs["raw_columns"] = raw
    return out

def split_list_col(series, dtype=int, sep=None):
    """'1 2 3' -> [1,2,3] for a whole column (paths, sequences)."""
    return series.map(lambda s: [dtype(x) for x in str(s).split(sep)])
