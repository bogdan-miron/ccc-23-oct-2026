"""Locate challenges/levels/inputs. Conventions seen in Cloudflight CCC Data&AI:
  <challenge>/level-N/{level-N.pdf, in_level-N_0-example.txt, out_level-N_0-example.txt, in_level-N_1-a.txt, ..., extras (*.in, *.zip)}
Outputs we write: submissions/<challenge>/level-N/out_level-N_K-name.<ext>  (same stem as the input, 'in_' -> 'out_')."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"cf", "tools", "tests", "knowledge", "solutions", "submissions", "notes", "notebooks", "_to_delete", ".claude", ".git", "__pycache__"}

def _lvl_no(p: Path):
    m = re.search(r"level[-_ ]?(\d+)", p.name, re.I)
    return int(m.group(1)) if m else None

class Level:
    def __init__(self, challenge: "Challenge", n: int, d: Path):
        self.ch, self.n, self.dir = challenge, n, d
    @property
    def pdf(self):
        c = sorted(self.dir.glob("*.pdf")); return c[0] if c else None
    @property
    def inputs(self):
        """in_* files, example first, then alphabetical by K."""
        fs = [f for f in self.dir.glob("in_*") if f.is_file()]
        if not fs:  # naming changed this year? fall back to every data file that is not a pdf/zip/out_
            fs = [f for f in self.dir.iterdir() if f.is_file() and f.suffix.lower() not in (".pdf", ".zip") and not f.name.startswith("out_")]
        return sorted(fs, key=lambda f: (not re.search(r"example", f.name, re.I), f.name))
    @property
    def extras(self):
        return sorted(f for f in self.dir.iterdir() if f.is_file() and not f.name.startswith(("in_", "out_")) and f.suffix.lower() != ".pdf")
    @property
    def out_dir(self):
        return ROOT / "submissions" / self.ch.name / f"level-{self.n}"
    def out_path(self, in_path):
        in_path = Path(in_path); return self.out_dir / re.sub(r"^in_", "out_", in_path.name)
    def expected(self, in_path):
        """Organizer-provided out_ file next to the input (examples only), else None."""
        e = self.dir / re.sub(r"^in_", "out_", Path(in_path).name); return e if e.exists() else None
    @property
    def notes_dir(self):
        return ROOT / "notes" / self.ch.name / f"level-{self.n}"
    def __repr__(self): return f"Level({self.ch.name}, {self.n}, inputs={[f.name for f in self.inputs]})"

class Challenge:
    def __init__(self, name_or_path):
        p = Path(name_or_path)
        self.dir = p if p.is_absolute() else ROOT / name_or_path
        if not self.dir.exists(): raise FileNotFoundError(self.dir)
        self.name = self.dir.name
    @property
    def module(self): return re.sub(r"\W", "_", self.name)
    def levels(self):
        ds = [d for d in self.dir.iterdir() if d.is_dir() and _lvl_no(d) is not None]
        return [Level(self, _lvl_no(d), d) for d in sorted(ds, key=_lvl_no)]
    def level(self, n):
        for l in self.levels():
            if l.n == int(n): return l
        raise KeyError(f"level {n} not found in {self.dir}")
    def prev_data(self, n):
        """Files the organizers hand over from earlier levels (all_data_from_level_*.in etc.) visible in level n."""
        return [f for f in self.level(n).extras if f.suffix in (".in", ".csv", ".txt", ".parquet")]

def find_challenges():
    return [Challenge(d) for d in sorted(ROOT.iterdir()) if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".") and any(_lvl_no(x) is not None for x in d.iterdir() if x.is_dir())]
