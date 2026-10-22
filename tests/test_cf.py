import numpy as np, pandas as pd, pytest
from pathlib import Path
from cf import io as cio, metrics, cv, models, submit

def test_word2num():
    assert cio.word2num("seventeen") == 17 and cio.word2num("twenty-one") == 21 and cio.word2num("minus five") == -5
    assert cio.word2num("one hundred five") == 105 and cio.word2num("siebzehn") == 17 and cio.word2num("abc") is None

def test_read_table_dirty(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("BOP,Temp [°C]\r\n1,20\r\n2,seventeen\r\n3,missing\r\n")
    df = cio.read_table(p); assert df["Temp [°C]"].tolist()[:2] == [20, 17] and np.isnan(df["Temp [°C]"].iloc[2])
    assert cio.snake("Bird Love Score [<3]") == "bird_love_score_lt3" and cio.snake("Temperature [°C]") == "temperature_c"

def test_topk():
    assert metrics.topk_overlap([[1, 2, 3]], [[3, 2, 9]], 3) == pytest.approx(2 / 3)

def test_time_split_no_leak():
    t = np.r_[np.arange(100)]; sp = cv.make_splits(100, kind="time", n_splits=4, time=t)
    for tr, va in sp: assert t[tr].max() < t[va].min()

def test_cv_and_bakeoff():
    rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd")); y = 3 * X.a - X.b + rng.normal(0, .1, 400)
    r = cv.cross_val(lambda: models.make("ridge", "reg"), X, y, "reg", verbose=False); assert r.score < 0.2
    yc = (X.a + X.b > 0).astype(int).map({0: "x", 1: "y"}).values
    r = cv.cross_val(lambda: models.make("lgbm", "clf", n_estimators=50), X, yc, "clf", verbose=False); assert r.score > 0.85
    try: import torch  # noqa: F401  (torch stage is optional: tabular-only contests never need it)
    except ImportError: return
    r = cv.cross_val(lambda: models.make("mlp", "clf", epochs=15), X, yc, "clf", verbose=False); assert r.score > 0.8
    r = cv.cross_val(lambda: models.make("lgbm", "clf", n_estimators=50), X, yc, "clf", metric="logloss", verbose=False); assert r.score < 0.5

def test_submit_style_and_check(tmp_path):
    ex = tmp_path / "ex.txt"; ex.write_bytes(b"BOP,Score [<3]\r\n3,28.20\r\n4,42.95\r\n")
    out = submit.write_text(tmp_path / "o.txt", "BOP,Score [<3]\n3,28.20\n4,42.95", like=ex)
    assert out.read_bytes() == ex.read_bytes() and submit.diff_example(out, ex)[0]
    assert submit.check_output(out, ex, expect_ids=[3, 4]) == []
    assert any("missing" in p for p in submit.check_output(out, ex, expect_ids=[3, 4, 5]))
    bad = submit.write_text(tmp_path / "b.txt", "BOP,Score\n3,nan"); assert submit.check_output(bad, ex)
