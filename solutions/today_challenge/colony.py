"""Colony: One engine + minimax (stdlib only). Rules from level-1.pdf and level-2.pdf.
Usage: python3 colony.py "cK:H2 cP:E2 mK:B5" [maxdepth] [round]
Finds the shortest forced CYAN win (CYAN to move) within maxdepth CYAN turns, against every MAGENTA reply.
Assumptions (flag if the site disagrees): guard choice with several guards = defender picks; Kernel-only check
happens at the end of every turn before the firewall; firewall rounds 8,16,24,26 remove ring 0,1,2,3."""
import sys, time
F = "ABCDEFGH"
def sq(s): return (F.index(s[0].upper()), int(s[1]) - 1)
def nm(p): return f"{F[p[0]]}{p[1]+1}"
ORTH = ((1, 0), (-1, 0), (0, 1), (0, -1)); DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
FW = {8: 0, 16: 1, 24: 2, 26: 3}
def cut_of(r): return 0 if r <= 8 else 1 if r <= 16 else 2 if r <= 24 else 3 if r <= 26 else 4
def ring(p): return min(p[0], p[1], 7 - p[0], 7 - p[1])
def inb(p, cut): return cut <= p[0] <= 7 - cut and cut <= p[1] <= 7 - cut
def key(b): return tuple(sorted((p[0], p[1], t, k) for p, (t, k) in b.items()))
def other(t): return "M" if t == "C" else "C"
def cheb(a, b): return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
def attack(b, a, t, team):
    tt, k = b[t]
    if k == "K":
        gs = sorted(p for p, (tm, _) in b.items() if tm == tt and p != t and cheb(p, t) == 1)
        if gs:
            for g in gs:
                nb = dict(b); del nb[g]; yield nb, None, f"x{nm(t)}(guard {nm(g)} dies)"
            return
        nb = dict(b); nb[t] = nb.pop(a); yield nb, team, f"x{nm(t)}"; return
    nb = dict(b); nb[t] = nb.pop(a); yield nb, None, f"x{nm(t)}"
def actions(b, team, cut):
    for pos, (tm, k) in list(b.items()):
        if tm != team: continue
        if k == "K":
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (pos[0] + dx, pos[1] + dy)
                    if not (dx or dy) or not inb(q, cut): continue
                    if q in b:
                        if b[q][0] == team: continue
                        for nb, w, d in attack(b, pos, q, team): yield nb, w, f"K{nm(pos)}{d}"
                    else:
                        nb = dict(b); nb[q] = nb.pop(pos); yield nb, None, f"K{nm(pos)}-{nm(q)}"
        else:
            for dx, dy in ORTH:
                for d in (1, 2):
                    q = (pos[0] + dx * d, pos[1] + dy * d)
                    if not inb(q, cut) or q in b: break
                    nb = dict(b); nb[q] = nb.pop(pos); yield nb, None, f"P{nm(pos)}-{nm(q)}"
            for dx, dy in DIAG:
                for d in (1, 2):
                    q = (pos[0] + dx * d, pos[1] + dy * d)
                    if not inb(q, cut): break
                    if q in b:
                        if b[q][0] != team:
                            for nb, w, dd in attack(b, pos, q, team): yield nb, w, f"P{nm(pos)}{dd}"
                        break
def turn(b, team, r):
    """All distinct outcomes of a full 2-action turn: {key: (desc, board, winner)} (end-of-turn rules applied)."""
    cut = cut_of(r); out = {}
    def fin(nb, w, desc):
        if w is None:
            if len(nb) == 2 and all(v[1] == "K" for v in nb.values()):
                ck = next(p for p, v in nb.items() if v[0] == "C"); mk = next(p for p, v in nb.items() if v[0] == "M")
                if cheb(ck, mk) <= 2: w = other(team)
                else:
                    a, c = ring(ck), ring(mk); w = "C" if a > c else "M"
        if w is None and team == "M" and r in FW:
            nb = {p: v for p, v in nb.items() if ring(p) != FW[r]}
            ck = any(v == ("C", "K") for v in nb.values()); mk = any(v == ("M", "K") for v in nb.values())
            if not ck: w = "M"
            elif not mk: w = "C"
        out.setdefault((w, key(nb)), (desc, nb, w))
    for b1, w1, d1 in actions(b, team, cut):
        if w1 is not None: fin(b1, w1, d1); continue
        for b2, w2, d2 in actions(b1, team, cut): fin(b2, w2, d1 + " ; " + d2)
    return list(out.values())
memo = {}
def cwin(b, r, d):
    """True if CYAN (to move, round r) can force a win within d cyan turns."""
    if d == 0 or r > 26: return False
    k = (key(b), r, d)
    if k in memo: return memo[k][0]
    best = None
    for desc, nb, w in sorted(turn(b, "C", r), key=lambda x: x[2] != "C"):
        if w == "C": memo[k] = (True, desc, nb); return True
        if w == "M": continue
        ok = True
        for _, mb, mw in turn(nb, "M", r):
            if mw == "M": ok = False; break
            if mw == "C": continue
            if not cwin(mb, r + 1, d - 1): ok = False; break
        if ok: memo[k] = (True, desc, nb); return True
    memo[k] = (False, None, None); return False
def parse(s):
    b = {}
    for t in s.split(): b[sq(t.split(":")[1])] = (t[0].upper(), t[1].upper())
    return b
def show(b):
    for y in range(7, -1, -1):
        print(f"{y+1} " + " ".join((b[(x, y)][0].lower() + b[(x, y)][1]) if (x, y) in b else " ." for x in range(8)))
    print("  " + " ".join(f" {c}" for c in F))
def main():
    b = parse(sys.argv[1]); maxd = int(sys.argv[2]) if len(sys.argv) > 2 else 4; r0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    t0 = time.time()
    for d in range(1, maxd + 1):
        if cwin(b, r0, d):
            _, desc, nb = memo[(key(b), r0, d)]
            print(f"FORCED WIN in {d} cyan turn(s). First turn: {desc}   [{time.time()-t0:.1f}s]"); show(nb)
            if d > 1:
                print("MAGENTA replies and the winning answer:")
                for mdesc, mb, mw in turn(nb, "M", r0):
                    if mw is None: 
                        mk = (key(mb), r0 + 1, d - 1)
                        cwin(mb, r0 + 1, d - 1); print(f"  {mdesc:40} -> {memo[mk][1]}")
            return
        print(f"no forced win within {d} turn(s) [{time.time()-t0:.1f}s]", flush=True)
if __name__ == "__main__": main()
