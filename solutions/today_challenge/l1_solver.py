"""Colony: One, level 1 (In a pinch). Stdlib only: python3 l1_solver.py "cK:D5 mK:E8 mP:D1"
Tokens: c|m (CYAN|MAGENTA) + K|P : square. Columns A-H left to right, rows 1-8 bottom to top (as on the site).
Lists every 2-action CYAN turn, grouped by final Kernel tile, with two safety levels:
  strict = MAGENTA has no 2-action turn that captures the CYAN Kernel (full MAGENTA turn)
  weak   = MAGENTA has no single action that captures it
Rules from the PDF: Kernel 1 step in 8 directions, captures by moving; Ping moves 1-2 tiles orthogonally onto empty
tiles (no jumping), captures the FIRST piece on a diagonal within 2 tiles if enemy. A turn is exactly 2 actions,
the same piece may act twice, no skipping (a turn ends early only if the enemy Kernel is captured)."""
import sys
F = "ABCDEFGH"
def sq(s): return (F.index(s[0].upper()), int(s[1]) - 1)
def nm(p): return f"{F[p[0]]}{p[1]+1}"
def inb(p): return 0 <= p[0] < 8 and 0 <= p[1] < 8
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1)); DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
def mv(b, a, c):
    n = dict(b); n[c] = n.pop(a); return n
def actions(b, team):
    for pos, (t, k) in list(b.items()):
        if t != team: continue
        if k == "K":
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (pos[0] + dx, pos[1] + dy)
                    if (dx or dy) and inb(q) and not (q in b and b[q][0] == team):
                        yield f"K {nm(pos)}{'x' if q in b else '-'}{nm(q)}", mv(b, pos, q)
        else:
            for dx, dy in ORTH:
                for d in (1, 2):
                    q = (pos[0] + dx * d, pos[1] + dy * d)
                    if not inb(q) or q in b: break
                    yield f"P {nm(pos)}-{nm(q)}", mv(b, pos, q)
            for dx, dy in DIAG:
                for d in (1, 2):
                    q = (pos[0] + dx * d, pos[1] + dy * d)
                    if not inb(q): break
                    if q in b:
                        if b[q][0] != team: yield f"P {nm(pos)}x{nm(q)}", mv(b, pos, q)
                        break
def has_k(b, team): return any(v == (team, "K") for v in b.values())
def turns(b, team):
    """All legal 2-action turns: (list of action strings, resulting board)."""
    for a1, b1 in actions(b, team):
        if not has_k(b1, "M" if team == "C" else "C"):
            yield [a1], b1; continue
        for a2, b2 in actions(b1, team): yield [a1, a2], b2
def main():
    b = {}
    for t in sys.argv[1].split():
        side, kind = t[0].upper(), t[1].upper(); b[sq(t.split(":")[1])] = (side, kind)
    ks = {}
    for seq, nb in turns(b, "C"):
        if not has_k(nb, "M"): print("WIN:", " ; ".join(seq)); continue
        weak = not any(not has_k(n1, "C") for _, n1 in actions(nb, "M"))
        strict = not any(not has_k(n2, "C") for _, n2 in turns(nb, "M"))
        kt = next(p for p, v in nb.items() if v == ("C", "K"))
        ks.setdefault(nm(kt), []).append((strict, weak, " ; ".join(seq)))
    print("final K tile | strict | weak | n seqs | example")
    for t in sorted(ks):
        L = ks[t]; s = all(x[0] for x in L); w = all(x[1] for x in L)
        print(f"{t:>12} | {str(s):6} | {str(w):5} | {len(L):6} | {L[0][2]}")
if __name__ == "__main__": main()
