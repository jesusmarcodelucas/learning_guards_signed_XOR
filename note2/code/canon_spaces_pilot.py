#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
canon_spaces_pilot.py -- companion code for
  "From experience to reusable rules II: event-centred canonicalisation by
   path integration across relational spaces"

One circuit, four relational spaces (line, ring, grid, binary tree).

  (M) signed mismatch per slot and code unit:  d = s' - s  ->  d+ and d-
  (A) anchor = gauge condition: onset of the change, found by the SAME signed comparison
      applied along the space (a changed slot whose predecessors are unchanged), then a
      winner-take-all with a fixed priority to break ties
  (P) path integration: a bump on a population of relative coordinates is RESET at the
      anchor and moved by one shift operator per generator of the space, gated by the
      salience map (it only enters changed slots)
  (C) coincidence: the code that left the hand (d- of the hand) must appear (d+) in every
      slot visited by the bump; then the identity is replaced by the role X
  (E) extent: the pattern traced by the bump is compared, with sign, to the shape code
      carried by the hand (written in the same relative coordinates)

Experiments (all numbers in the note are printed by this script):
  1. staircase raw -> select -> relocate -> substitute -> bind, per space
  2. ablations of each piece of the circuit, per space
  3. reuse: the effect, bound to withheld colours and withheld shapes, written by the same
     integrator reset at a proposed target, on larger unseen environments
  4. choosing the frame from data: four candidate frames, fragmentation vs collision
  5. content frames (gauge-like): per-slot codebooks, transport along the bump path,
     a curved link, Wilson loops and path dependence

Only NumPy. Fixed seeds. Run:  python3 canon_spaces_pilot.py
"""
import json, zlib, numpy as np
from collections import deque, Counter

# ------------------------------------------------------------------------------------------
# Filler codebook: identical to circuit_pilot.py of note I (6 colours, 6 active bits of 64)
# ------------------------------------------------------------------------------------------
N, KC, NCOL = 64, 6, 6
_rc = np.random.default_rng(5)
CODE = np.zeros((NCOL + 1, N), dtype=int)            # row 0 = empty (all zeros)
for c in range(1, NCOL + 1):
    CODE[c, _rc.choice(N, KC, replace=False)] = 1
TRAIN_COLOURS, TEST_COLOURS = (1, 2, 3), (4, 5, 6)

# ------------------------------------------------------------------------------------------
# Spaces: nodes (slots) and generators (labelled partial maps node -> node)
# ------------------------------------------------------------------------------------------
class Space:
    """A relational space. gens[g][i] = node reached from i by generator g, or -1.
    inv[g][j]  = node i with gens[g][i] = j, or -1 (the predecessor along g).
    prio[i]    = fixed priority used by the winner-take-all (smaller wins).
    algebra    = which relative-coordinate population the integrator uses."""
    def __init__(self, kind, size):
        self.kind, self.size = kind, size
        if kind == "line":
            L = size; self.n = L
            self.gens = {"r": [i + 1 if i + 1 < L else -1 for i in range(L)]}
            self.algebra = "Z1"; self.prio = list(range(L))
        elif kind == "ring":
            L = size; self.n = L
            self.gens = {"r": [(i + 1) % L for i in range(L)]}
            self.algebra = "Z1"; self.prio = list(range(L))
        elif kind == "grid":
            H, W = size; self.n = H * W; self.H, self.W = H, W
            self.gens = {"r": [i + 1 if (i % W) + 1 < W else -1 for i in range(self.n)],
                         "d": [i + W if i + W < self.n else -1 for i in range(self.n)]}
            self.algebra = "Z2"; self.prio = list(range(self.n))      # row-major order
        elif kind == "tree":
            D = size; self.n = 2 ** D - 1
            self.gens = {"a": [2 * i + 1 if 2 * i + 1 < self.n else -1 for i in range(self.n)],
                         "b": [2 * i + 2 if 2 * i + 2 < self.n else -1 for i in range(self.n)]}
            self.algebra = "W2"; self.prio = list(range(self.n))      # heap order: depth, then left
        self.gens = {g: np.array(v) for g, v in self.gens.items()}
        self.inv = {}
        for g, f in self.gens.items():
            iv = -np.ones(self.n, dtype=int)
            for i, j in enumerate(f):
                if j >= 0: iv[j] = i
            self.inv[g] = iv
        # moves used by the bump: each generator and its inverse, in a fixed order
        self.moves = []
        for g in self.gens:
            self.moves += [(g, +1), (g, -1)]

    def step(self, i, g, s):
        return int(self.gens[g][i]) if s > 0 else int(self.inv[g][i])

    def label(self):
        return f"{self.kind}{self.size}"

# ------------------------------------------------------------------------------------------
# Relative-coordinate populations and their shift operators (one per generator)
# ------------------------------------------------------------------------------------------
class Integrator:
    """A finite population of relative coordinates. The bump is a one-hot vector over it;
    generator g moves it by the (partial) permutation matrix S[g, +1] and S[g, -1].
    Z1: offsets -K..K (line, ring).  Z2: (row, col) offsets in -K..K (grid).
    W2: words over {a, b} up to depth K, coded as heap indices (tree)."""
    def __init__(self, algebra, K):
        self.algebra = algebra
        if algebra == "Z1":
            self.coords = [(x,) for x in range(-K, K + 1)]
            shift = {("r", +1): lambda c: (c[0] + 1,), ("r", -1): lambda c: (c[0] - 1,)}
            self.origin = (0,)
        elif algebra == "Z2":
            self.coords = [(y, x) for y in range(-K, K + 1) for x in range(-K, K + 1)]
            shift = {("r", +1): lambda c: (c[0], c[1] + 1), ("r", -1): lambda c: (c[0], c[1] - 1),
                     ("d", +1): lambda c: (c[0] + 1, c[1]), ("d", -1): lambda c: (c[0] - 1, c[1])}
            self.origin = (0, 0)
        elif algebra == "W2":
            M = 2 ** (K + 1) - 1
            self.coords = [(j,) for j in range(M)]
            def par(c, side):
                j = c[0]
                if j == 0: return None
                if side == "a" and j % 2 == 1: return ((j - 1) // 2,)
                if side == "b" and j % 2 == 0: return ((j - 2) // 2,)
                return None
            shift = {("a", +1): lambda c: (2 * c[0] + 1,), ("b", +1): lambda c: (2 * c[0] + 2,),
                     ("a", -1): lambda c: par(c, "a"), ("b", -1): lambda c: par(c, "b")}
            self.origin = (0,)
        self.index = {c: k for k, c in enumerate(self.coords)}
        self.M = len(self.coords)
        self.S = {}
        for key, f in shift.items():
            S = np.zeros((self.M, self.M), dtype=int)
            for c, k in self.index.items():
                t = f(c)
                if t is not None and t in self.index: S[self.index[t], k] = 1
            self.S[key] = S

    def reset(self):
        b = np.zeros(self.M, dtype=int); b[self.index[self.origin]] = 1; return b

    def coord(self, b):
        return self.coords[int(np.argmax(b))]

    def word_coord(self, word):
        """Integrate a word of moves from the origin; returns the coordinate or None."""
        b = self.reset()
        for g, s in word:
            b = self.S[(g, s)] @ b
            if not b.any(): return None
        return self.coord(b)

INTEG = {"Z1": Integrator("Z1", 6), "Z2": Integrator("Z2", 4), "W2": Integrator("W2", 4)}

# ------------------------------------------------------------------------------------------
# Shapes carried by the hand: words of moves from the shape's own anchor
# ------------------------------------------------------------------------------------------
R, L_, D, U, A, B = ("r", +1), ("r", -1), ("d", +1), ("d", -1), ("a", +1), ("b", +1)
SHAPES = {
    "line": {"w1": [()], "w2": [(), (R,)], "w3": [(), (R,), (R, R)]},
    "ring": {"w1": [()], "w2": [(), (R,)], "w3": [(), (R,), (R, R)]},
    "grid": {"mono": [()], "hdom": [(), (R,)], "vdom": [(), (D,)],
             "Lda": [(), (D,), (D, R)], "Lrd": [(), (R,), (R, D)], "Lrv": [(), (R,), (D,)],
             "Ldl": [(), (D,), (D, L_)], "Ih": [(), (R,), (R, R)], "Iv": [(), (D,), (D, D)],
             "sq": [(), (R,), (D,), (R, D)]},
    "tree": {"t1": [()], "ta": [(), (A,)], "tb": [(), (B,)],
             "fork": [(), (A,), (B,)], "chain": [(), (A,), (A, A)]},
}
TRAIN_SHAPES = {"line": ["w1", "w2"], "ring": ["w1", "w2"],
                "grid": ["mono", "hdom", "vdom"], "tree": ["t1", "ta", "tb"]}
TEST_SHAPES = {"line": ["w3"], "ring": ["w3"],
               "grid": ["Lda", "Lrd", "Lrv", "Ldl", "Ih", "Iv", "sq"], "tree": ["fork", "chain"]}
TRAIN_ENVS = {"line": [4, 5, 6], "ring": [5, 6, 7], "grid": [(3, 3), (3, 4), (4, 4)], "tree": [3, 4]}
TEST_ENVS = {"line": [4, 5, 6, 7, 8], "ring": [5, 6, 7, 8, 9],
             "grid": [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)], "tree": [3, 4, 5]}

def shape_pattern(kind, shape):
    """The hand's shape code: the set of relative coordinates of the shape (one-hot union)."""
    I = INTEG[Space(kind, TRAIN_ENVS[kind][0]).algebra]
    v = np.zeros(I.M, dtype=int)
    for w in SHAPES[kind][shape]:
        v[I.index[I.word_coord(w)]] = 1
    return v

def footprint(space, t, shape):
    """World rule: nodes covered by `shape` placed with its anchor at node t (None if off-space)."""
    out = []
    for word in SHAPES[space.kind][shape]:
        i = t
        for g, s in word:
            i = space.step(i, g, s)
            if i < 0: return None
        out.append(i)
    return out

# ------------------------------------------------------------------------------------------
# The world: random placement episodes with persistent identities
# ------------------------------------------------------------------------------------------
def play(kind, episodes, rng, colours=TRAIN_COLOURS, shapes=None):
    shapes = shapes or TRAIN_SHAPES[kind]
    events = []
    for _ in range(episodes):
        sp = Space(kind, TRAIN_ENVS[kind][rng.integers(len(TRAIN_ENVS[kind]))])
        ids = np.zeros(sp.n, dtype=int)
        while True:
            c = int(colours[rng.integers(len(colours))])
            order = list(rng.permutation(len(shapes)))
            placed = False
            for k in order:
                sh = shapes[k]
                legal = []
                for t in range(sp.n):
                    fp = footprint(sp, t, sh)
                    if fp is not None and not ids[fp].any(): legal.append(fp)
                if legal:
                    fp = legal[rng.integers(len(legal))]
                    after = ids.copy(); after[fp] = c
                    events.append(dict(space=sp, before=ids, after=after, c=c, shape=sh))
                    ids = after; placed = True; break
            if not placed: break
    return events

# ------------------------------------------------------------------------------------------
# The circuit
# ------------------------------------------------------------------------------------------
def mismatch(before, after, c):
    S0, S1 = CODE[before], CODE[after]
    d = S1 - S0
    return (d > 0).astype(int), (d < 0).astype(int), CODE[c]      # hand: colour -> nothing

def anchor_of(space, sal, rule="onset"):
    """Gauge condition. 'onset': a changed slot none of whose predecessors (along any
    generator) is changed -- the d+ channel of the signed comparison of salience along the
    space -- then winner-take-all by fixed priority. 'index': WTA by node index directly."""
    idx = np.flatnonzero(sal)
    if len(idx) == 0: return None
    if rule == "onset":
        pred_on = np.zeros(space.n, dtype=int)
        for g in space.gens:
            p = space.inv[g]
            pred_on |= np.where(p >= 0, sal[np.maximum(p, 0)], 0)
        onset = sal * (1 - pred_on)                     # (1 - a) * b : the d+ channel
        cand = np.flatnonzero(onset)
        if len(cand) == 0: cand = idx                   # no onset (e.g. a full ring): fall back
    else:
        cand = idx
    return int(min(cand, key=lambda i: space.prio[i]))

def integrate(space, sal, anchor, moves=None):
    """Path integration from the anchor: BFS over changed slots; the bump is reset at the
    anchor and moved by the shift operator of each generator taken. Returns {node: bump}."""
    I = INTEG[space.algebra]
    moves = moves or space.moves
    bump = {anchor: I.reset()}; parent = {anchor: None}
    q = deque([anchor])
    while q:
        u = q.popleft()
        for g, s in moves:
            v = space.step(u, g, s)
            if v < 0 or not sal[v] or v in bump: continue
            b = I.S[(g, s)] @ bump[u]
            if not b.any(): continue                    # off the attractor's range
            bump[v] = b; parent[v] = (u, g, s); q.append(v)
    return bump, parent

def canonical(ev, reset=True, anchor_rule="onset", bind_role=True, bind_extent=True,
              integrate_moves=None, content=None):
    """Run the circuit on one accepted transition. Returns (pattern, info).
    `content`: optional dict for the gauge experiment (per-slot code frames and links)."""
    sp = ev["space"]; I = INTEG[sp.algebra]
    if content is None:
        dplus, dminus, hminus = mismatch(ev["before"], ev["after"], ev["c"])
    else:
        dplus, dminus, hminus = content["dplus"], content["dminus"], content["hminus"]
    sal = ((dplus + dminus).sum(axis=1) > 0).astype(int)
    a = anchor_of(sp, sal, anchor_rule)
    bump, parent = integrate(sp, sal, a, integrate_moves)
    visited = sorted(bump, key=lambda i: sp.prio[i])
    T = np.zeros(I.M, dtype=int)
    for v in visited: T |= bump[v]
    residual = int(sal.sum() - len(visited))
    # (C) coincidence of the hand's d- code with the d+ code under the bump
    if content is None:
        carried = {v: hminus for v in visited}
    else:
        carried = content["transport"](sp, a, parent, visited)
    Xr = {v: int(dplus[v] @ carried[v] >= KC - 1 and not dminus[v].any()) for v in visited}
    role_ok = all(Xr.values())
    # (E) extent: signed comparison of traced pattern and hand shape code
    Hs = shape_pattern(sp.kind, ev["shape"])
    dE = T - Hs
    ext_ok = not dE.any()
    hand_empty = True                                   # every accepted event empties the hand
    info = dict(anchor=a, visited=visited, residual=residual, role=role_ok, ext=ext_ok,
                coords={v: I.coord(bump[v]) for v in visited})
    if reset:
        geom = tuple(sorted(info["coords"].values()))
    else:
        geom = (sp.label(),) + tuple(visited)
    if bind_role:
        who = ("X",) if role_ok else ("noX",) + tuple(sorted(Xr.items()))
    else:
        who = tuple(np.flatnonzero(hminus).tolist())
    if bind_extent and reset and ext_ok and residual == 0 and hand_empty:
        if bind_role and role_ok:
            return ("DELTA: X fills the hand's shape from the anchor; hand <- empty",), info
        if not bind_role:      # isolated ablation: extent still bound, the identity stays in the pattern
            return ("DELTA with the identity kept: this colour fills the hand's shape; hand <- empty",) + who, info
    shape_tag = () if bind_extent else (ev["shape"],)
    return (geom, who, int(ext_ok), residual) + shape_tag, info

def engram(pattern, _cache={}):
    """Fixed sparse expansion of a pattern (random projection + top-16), as in note I.
    Here patterns are hashed into a 32-dim real vector first, since their length varies."""
    h = np.random.default_rng(zlib.crc32(repr(pattern).encode())).standard_normal(32)
    if "P" not in _cache: _cache["P"] = np.random.default_rng(7).standard_normal((512, 32))
    return frozenset(np.argsort(_cache["P"] @ h)[-16:].tolist())

# ------------------------------------------------------------------------------------------
# Applying the effect: the same integrator, reset at a PROPOSED target, gated by the shape
# ------------------------------------------------------------------------------------------
def free_trace(space, t, shape, ids):
    """P gated by 'free': move the bump from t along the shape's words while slots are free.
    Returns the traced pattern (relative coords that are free and exist)."""
    I = INTEG[space.algebra]; v = np.zeros(I.M, dtype=int)
    for word in SHAPES[space.kind][shape]:
        i, b, ok = t, I.reset(), True
        for g, s in word:
            i = space.step(i, g, s); b = I.S[(g, s)] @ b
            if i < 0 or not b.any(): ok = False; break
        if ok and ids[i] == 0: v |= b
    return v

def write_delta(space, t, shape, ids, colour):
    """Write X (bound to `colour`) at every node reached by the bump from t along the shape."""
    I = INTEG[space.algebra]; pred = ids.copy()
    for word in SHAPES[space.kind][shape]:
        i = t
        for g, s in word:
            i = space.step(i, g, s)
        pred[i] = colour
    return pred

# ------------------------------------------------------------------------------------------
# Frames for experiment 4
# ------------------------------------------------------------------------------------------
def frame_key(ev, frame):
    """Canonical pattern of the CHANGED geometry under a candidate frame (roles substituted,
    extent not bound), plus the decoder to reconstruct the change at another event."""
    sp = ev["space"]
    ch = np.flatnonzero(ev["before"] != ev["after"])
    if frame == "absolute":
        return (sp.label(), tuple(ch.tolist()))
    a_idx = int(ch.min())
    if frame == "index_offset":
        return tuple((ch - a_idx).tolist())
    if frame == "count":
        return (len(ch),)
    if frame == "generators":
        pat, info = canonical(ev, bind_extent=False)
        return pat[0]

def frame_decode(ev, frame, key):
    sp = ev["space"]
    ch = np.flatnonzero(ev["before"] != ev["after"])
    a_idx = int(ch.min())
    if frame == "absolute":
        return set(key[1]) if key[0] == sp.label() else None
    if frame == "index_offset":
        nodes = [a_idx + o for o in key]
        return set(nodes) if all(0 <= x < sp.n for x in nodes) else None
    if frame == "count":
        nodes = list(range(a_idx, a_idx + key[0]))
        return set(nodes) if nodes[-1] < sp.n else None
    if frame == "generators":
        # walk the whole space from the onset anchor with the integrator (no gating)
        sal = (ev["before"] != ev["after"]).astype(int)
        a = anchor_of(sp, sal, "onset")
        bump, _ = integrate(sp, np.ones(sp.n, dtype=int), a)
        I = INTEG[sp.algebra]
        inv = {I.coord(b): v for v, b in bump.items()}
        nodes = [inv.get(c) for c in key]
        return set(nodes) if None not in nodes else None

# ------------------------------------------------------------------------------------------
# Content frames (gauge-like experiment)
# ------------------------------------------------------------------------------------------
def make_frames(sp, rng, curved_edge=None):
    """Per-slot codebooks: slot i stores colour c as P_i(code_c), P_i a random permutation
    of the 64 units; the hand uses P_h. Links along generators: L[u,v] = P_v P_u^{-1}
    (pure gauge). With `curved_edge` = (u, g), that link is multiplied by a random
    permutation sigma (and its reverse by sigma^{-1}): a local curvature."""
    perm = {i: rng.permutation(N) for i in range(sp.n)}; perm["h"] = rng.permutation(N)
    invp = {k: np.argsort(p) for k, p in perm.items()}
    link = {}
    for g in sp.gens:
        for u in range(sp.n):
            v = int(sp.gens[g][u])
            if v < 0: continue
            link[(u, v)] = invp[u][perm[v]]               # x -> x[link]  ==  P_v P_u^{-1}
            link[(v, u)] = invp[v][perm[u]]
    if curved_edge is not None:
        u, g = curved_edge; v = int(sp.gens[g][u])
        sigma = rng.permutation(N)
        link[(u, v)] = link[(u, v)][sigma]
        link[(v, u)] = np.argsort(link[(u, v)])           # keep the reverse link consistent
    return perm, invp, link

def local_code(perm, i, colour):
    return CODE[colour][perm[i]]

def wilson_loops(sp, link):
    """Product of links around every elementary square (r, d, -r, -d); count non-trivial."""
    nontriv = 0; total = 0
    for u in range(sp.n):
        v = int(sp.gens["r"][u]); w = int(sp.gens["d"][u])
        if v < 0 or w < 0: continue
        x = int(sp.gens["d"][v]); total += 1
        idx = np.arange(N)
        for (p, q) in [(u, v), (v, x), (x, w), (w, u)]:
            idx = idx[link[(p, q)]]
        nontriv += int(not np.array_equal(idx, np.arange(N)))
    return nontriv, total

def gauge_content(ev, perm, invp, link, transport=True):
    sp = ev["space"]
    S0 = np.stack([local_code(perm, i, ev["before"][i]) for i in range(sp.n)])
    S1 = np.stack([local_code(perm, i, ev["after"][i]) for i in range(sp.n)])
    d = S1 - S0
    hminus = local_code(perm, "h", ev["c"])
    def tr(space, a, parent, visited):
        out = {}
        for v in visited:
            path = []; x = v
            while parent[x] is not None:
                path.append((parent[x][0], x)); x = parent[x][0]
            code = hminus[invp["h"][perm[a]]] if transport else hminus   # hand -> anchor link
            if transport:
                for (p, q) in reversed(path): code = code[link[(p, q)]]
            out[v] = code
        return out
    return dict(dplus=(d > 0).astype(int), dminus=(d < 0).astype(int), hminus=hminus, transport=tr)

# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    out = {}
    KINDS = ["line", "ring", "grid", "tree"]
    EV = {k: play(k, 400, np.random.default_rng(3)) for k in KINDS}

    print("1. STAIRCASE (distinct event descriptions after each step), 400 episodes per space")
    print(f"   {'space':6s} {'events':>7s} {'raw':>6s} {'select':>7s} {'relocate':>9s} {'substitute':>11s} {'bind':>5s}  validated  engrams")
    stair = {}
    for k in KINDS:
        E = EV[k]
        raw = set(); sel = set(); rel = set(); sub = set(); bnd = set(); eng = set(); ok = 0
        for ev in E:
            sp = ev["space"]
            ch = np.flatnonzero(ev["before"] != ev["after"])
            raw.add((sp.label(), tuple(ev["before"]), tuple(ev["after"]), ev["c"], ev["shape"]))
            sel.add((sp.label(), tuple((int(i), int(ev["before"][i]), int(ev["after"][i])) for i in ch), ev["c"], ev["shape"]))
            pat, info = canonical(ev)
            rel.add((tuple(sorted((info["coords"][v], int(ev["before"][v]), int(ev["after"][v])) for v in info["visited"])), ev["c"], ev["shape"]))
            sub.add((tuple(sorted((info["coords"][v], "empty", "X") for v in info["visited"])), "X", ev["shape"]))
            bnd.add(pat); eng.add(engram(pat))
            ok += int(info["role"] and info["ext"] and info["residual"] == 0)
        stair[k] = dict(events=len(E), raw=len(raw), select=len(sel), relocate=len(rel),
                        substitute=len(sub), bind=len(bnd), validated=ok, engrams=len(eng))
        s = stair[k]
        print(f"   {k:6s} {s['events']:7d} {s['raw']:6d} {s['select']:7d} {s['relocate']:9d} {s['substitute']:11d} {s['bind']:5d}  {ok:5d}/{len(E):<5d} {s['engrams']:5d}")
    out["staircase"] = stair

    print("\n2. ABLATIONS (distinct patterns over the same events)")
    abl = {}
    for k in KINDS:
        E = EV[k]; row = {}
        for name, kw in [("full", {}), ("no_reset", dict(reset=False)), ("index_anchor", dict(anchor_rule="index")),
                         ("no_role", dict(bind_role=False)), ("no_extent", dict(bind_extent=False))]:
            pats = [canonical(ev, **kw)[0] for ev in E]
            row[name] = len(set(pats))
            assert len(set(engram(p) for p in pats)) == row[name]      # engrams = patterns
        abl[k] = row
        print(f"   {k:6s} " + "  ".join(f"{n}={v}" for n, v in row.items()))
    # which ring events break under the index anchor?
    wrap = sum(1 for ev in EV["ring"] if canonical(ev, anchor_rule="index")[0] != canonical(ev)[0])
    abl["ring_wrapped_events"] = wrap
    print(f"   ring events whose description changes with the index anchor (runs crossing the seam): {wrap}/{len(EV['ring'])}")
    # a full-ring event: no onset exists; the fallback still yields the canonical description
    sp = Space("ring", 4); evf = dict(space=sp, before=np.zeros(4, int), after=np.full(4, 2), c=2, shape="w1")
    pf, inf = canonical(evf, bind_extent=False)
    abl["full_ring_example"] = dict(anchor=inf["anchor"], coords=sorted(inf["coords"].values()))
    print(f"   full ring (4 slots, all change): no onset; fallback anchor {inf['anchor']}; traced coords {sorted(inf['coords'].values())}")
    # around the whole ring the integer integrator is path dependent (winding = holonomy)
    hol = {}
    for name, mv in [("right_first", [("r", 1), ("r", -1)]), ("left_first", [("r", -1), ("r", 1)])]:
        b, _ = integrate(sp, np.ones(4, dtype=int), 0, mv)
        hol[name] = {v: INTEG["Z1"].coord(x)[0] for v, x in sorted(b.items())}
    abl["full_ring_coords_by_order"] = hol
    print(f"   same full ring, integer coordinates by traversal order: {hol}  (slot 2: +2 or -2; equal mod 4)")
    out["ablations"] = abl

    print("\n3. REUSE: withheld colours 4-6, withheld shapes, larger environments")
    rng = np.random.default_rng(17); reuse = {}
    for k in KINDS:
        n_cases = n_perm = fp_ = fr_ = werr = rterr = 0; n_new = 0
        for env in TEST_ENVS[k]:
            sp = Space(k, env)
            if sp.n <= 9: states = [np.array([(b >> i) & 1 for i in range(sp.n)]) for b in range(2 ** sp.n)]
            else: states = [rng.integers(0, 2, sp.n) for _ in range(600)]
            for occ in states:
                ids = occ * int(rng.integers(1, 4))           # occupied slots carry some identity
                for sh in TRAIN_SHAPES[k] + TEST_SHAPES[k]:
                    Hs = shape_pattern(k, sh)
                    for t in range(sp.n):
                        n_cases += 1
                        fp = footprint(sp, t, sh)
                        world_ok = fp is not None and not ids[fp].any()
                        neg = (free_trace(sp, t, sh, ids) - Hs) < 0           # N- : missing cells
                        permit = not neg.any()
                        fp_ += int(permit and not world_ok); fr_ += int(world_ok and not permit)
                        if not permit: continue
                        n_perm += 1; n_new += int(sh in TEST_SHAPES[k])
                        c = int(TEST_COLOURS[rng.integers(3)])
                        pred = write_delta(sp, t, sh, ids, c)
                        after = ids.copy(); after[fp] = c
                        werr += int(not np.array_equal(pred, after))
                        pat, _ = canonical(dict(space=sp, before=ids, after=pred, c=c, shape=sh))
                        rterr += int(pat != canonical(EV[k][0])[0])
        reuse[k] = dict(cases=n_cases, permitted=n_perm, permitted_new_shapes=n_new,
                        false_permissions=fp_, false_refusals=fr_, write_errors=werr, recognition_mismatch=rterr)
        print(f"   {k:6s} cases {n_cases:7d}; permitted {n_perm:6d} (new shapes {n_new:6d}); false perm {fp_}; "
              f"false ref {fr_}; write errors {werr}; re-recognised as another engram {rterr}")
    out["reuse"] = reuse

    print("\n4. CHOOSING THE FRAME: distinct patterns (fragmentation) and reconstruction errors (collision)")
    print("   roles substituted, extent NOT bound; ideal = number of training shapes, 0 errors")
    FR = ["absolute", "index_offset", "count", "generators"]; fsel = {}
    for k in KINDS:
        row = {}
        for f in FR:
            proto = {}; err = 0
            for ev in EV[k]:
                key = frame_key(ev, f)
                if key not in proto: proto[key] = ev
                rec = frame_decode(ev, f, key)
                true = set(np.flatnonzero(ev["before"] != ev["after"]).tolist())
                err += int(rec != true)
            row[f] = dict(patterns=len(proto), errors=err)
        valid = [f for f in FR if row[f]["errors"] == 0]
        best = min(row[f]["patterns"] for f in valid)
        row["selected"] = [f for f in valid if row[f]["patterns"] == best]
        row["n_train_shapes"] = len(TRAIN_SHAPES[k])
        fsel[k] = row
        print(f"   {k:6s} " + "  ".join(f"{f}: {row[f]['patterns']}/{row[f]['errors']}" for f in FR)
              + f"   -> selected {row['selected']} (shapes {row['n_train_shapes']})")
    out["frames"] = fsel

    print("\n5. CONTENT FRAMES (gauge-like): grid world, per-slot codebooks")
    rng = np.random.default_rng(23)
    Eg = play("grid", 400, np.random.default_rng(29), shapes=["mono", "hdom", "vdom", "sq", "Lrd"])
    gauge = {}
    base = set(canonical(ev)[0] for ev in Eg)
    gauge["events"] = len(Eg); gauge["shared_codebook_patterns"] = len(base)
    frames = {}
    for ev in Eg:
        lab = ev["space"].label()
        if lab not in frames:
            sp = ev["space"]
            mid = (sp.H // 2) * sp.W + (sp.W // 2) - 1                # an interior 'r' link
            frames[lab] = dict(flat=make_frames(sp, rng), curved=make_frames(sp, rng, curved_edge=(mid, "r")),
                               curved_edge=(mid, int(sp.gens["r"][mid])))
    res = {}
    for name, which, tr in [("local_no_transport", "flat", False), ("local_transport_flat", "flat", True),
                            ("local_transport_curved", "curved", True)]:
        pats = []; okc = 0
        for ev in Eg:
            perm, invp, link = frames[ev["space"].label()][which]
            ct = gauge_content(ev, perm, invp, link, transport=tr)
            p, inf = canonical(ev, content=ct); pats.append(p); okc += int(inf["role"])
        res[name] = dict(patterns=len(set(pats)), role_bound=okc)
    # path dependence under curvature: two orders of the generators
    ordA = [("r", 1), ("r", -1), ("d", 1), ("d", -1)]; ordB = [("d", 1), ("d", -1), ("r", 1), ("r", -1)]
    diff = 0; diff_sq = 0; nsq = 0; flatdiff = 0
    for ev in Eg:
        for which in ("flat", "curved"):
            perm, invp, link = frames[ev["space"].label()][which]
            ct = gauge_content(ev, perm, invp, link, transport=True)
            pa, ia = canonical(ev, content=ct, integrate_moves=ordA)
            pb, ib = canonical(ev, content=ct, integrate_moves=ordB)
            if which == "flat": flatdiff += int(ia["role"] != ib["role"])
            else:
                diff += int(ia["role"] != ib["role"])
                if ev["shape"] == "sq": nsq += 1; diff_sq += int(ia["role"] != ib["role"])
    res["order_dependence_flat"] = flatdiff; res["order_dependence_curved"] = diff
    # squares that cover a curved elementary square: is the curved link their top or bottom edge?
    split = dict(top_edge=[0, 0], bottom_edge=[0, 0])       # [placements, order-dependent]
    crossing = 0
    for ev in Eg:
        f = frames[ev["space"].label()]; u, v = f["curved_edge"]
        sal = (ev["before"] != ev["after"]).astype(int)
        a = anchor_of(ev["space"], sal); _, par = integrate(ev["space"], sal, a)
        used = {(par[x][0], x) for x in par if par[x] is not None}
        crossing += int((u, v) in used or (v, u) in used)
        if ev["shape"] != "sq" or not (sal[u] and sal[v]): continue
        perm, invp, link = f["curved"]; ct = gauge_content(ev, perm, invp, link, transport=True)
        ra = canonical(ev, content=ct, integrate_moves=ordA)[1]["role"]
        rb = canonical(ev, content=ct, integrate_moves=ordB)[1]["role"]
        key = "top_edge" if u == a else "bottom_edge"
        split[key][0] += 1; split[key][1] += int(ra != rb)
    res["squares_on_curved_link"] = split; res["bump_paths_crossing_curved_link"] = crossing
    res["order_dependence_curved_squares"] = [diff_sq, nsq]
    wl = {}
    for lab, f in frames.items():
        H, W = [int(x) for x in lab[5:-1].split(", ")]
        sp = Space("grid", (H, W))
        wl[lab] = dict(flat=wilson_loops(sp, f["flat"][2]), curved=wilson_loops(sp, f["curved"][2]))
    res["wilson_loops_nontrivial_of_total"] = wl
    gauge.update(res)
    print(f"   {len(Eg)} events (shapes mono, hdom, vdom, square, L); shared codebook: {len(base)} pattern(s)")
    for name in ("local_no_transport", "local_transport_flat", "local_transport_curved"):
        print(f"   {name:24s} patterns {res[name]['patterns']:4d}; X bound in {res[name]['role_bound']}/{len(Eg)}")
    print(f"   order dependence (r-first vs d-first BFS): flat {flatdiff}, curved {diff} (squares {diff_sq}/{nsq})")
    print(f"   bump paths crossing the curved link: {crossing}; squares covering it [placements, order-dependent]: {split}")
    print(f"   non-trivial Wilson loops (of all plaquettes): {wl}")
    out["gauge"] = gauge

    with open("numbers_note2.json", "w") as f: json.dump(out, f, indent=1, default=str)
    print("\nwritten numbers_note2.json")
