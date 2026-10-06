#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heteroclinic_coordinate_pilot.py -- companion to note II, Section "Relation to heteroclinic execution".

Question: can the path-integration primitive P of note II -- the bump that is reset at an
anchor and moved one relative coordinate at a time -- be a heteroclinic sequence, so that
the breadth-first scheduler of canon_spaces_pilot.py is replaced by dynamics? The sequence
0 -> 1 -> 2 -> ... is INTRINSIC to the asymmetric connectivity; the event only resets it
(initial condition at the anchor) and STOPS it (an inhibitory input where the event ends).

Construction (line and ring only; no branching):
  * one population per relative coordinate r = 0, 1, ..., M-1, generalised Lotka-Volterra
        dx_r/dt = x_r ( 1 - u_r - sum_s rho[r,s] x_s ) + eps + noise
    with rho[r,r] = 1, rho[r+1,r] = 0.5 (the successor can invade: lambda = 0.5),
    rho = 2 otherwise (predecessors and the rest are suppressed). For eps = 0 and u = 0 each
    x_r = 1 is a saddle and the chain 0 -> 1 -> 2 -> ... is a heteroclinic sequence; with the
    small eps > 0 used here the exact equilibria are removed and trajectories follow the
    heteroclinic channel with finite dwell times.
  * the reset at the anchor is the initial condition x_0 = 1;
  * the event STOPS the sequence with an inhibitory input u_r in [0, 1] at the first
    population beyond the event: the escape from r has rate 1 - u_{r+1} - rho[r+1,r],
    so it is removed if u_{r+1} > 1 - rho[r+1,r] = 0.5; the
    trajectory ends near r (H6 sweeps u);
  * recognition: u_r = 1 where slot anchor + r did NOT change; execution: u_r = 1 outside the
    hand's width code (r >= w); guard: u_r = 1 also where slot anchor + r is missing or full;
  * coordinate populations: an open INTEGER chain (r = 0..M-1), or, on a ring of L slots,
    a heteroclinic CYCLE of L populations (rho[0, L-1] = 0.5), which is Z_L by construction.

Tests:
  H1 recognition of the note-II training placements (line, ring) by the heteroclinic P,
     compared event by event with the breadth-first construction
  H2 execution and guard: write Delta and compute N- with the same chain, exhaustive states
  H3 an event that covers a whole ring: integer chain vs integer chain with inhibition of
     return vs heteroclinic cycle of L populations (ring size known)
  H4 timing: dwell per saddle vs eps (fitted slope vs 1/lambda), traversal time vs extent
  H5 noise scan: 100 independent noise realisations x extents 1..6 x 7 noise levels
     (correct sequence, order errors, premature stops, overruns, dwell mean and CV), plus the
     training placements; figure fig_noise_scan.pdf
  H6 stop strength: the sequence ends iff u > 1 - rho[r+1,r] = 0.5

Only NumPy. Fixed seeds. Run from code/:  python3 heteroclinic_coordinate_pilot.py
"""
import json, numpy as np
import canon_spaces_pilot as cs

DT, LAM = 0.05, 0.5

def make_rho(M, cycle=False):
    rho = np.full((M, M), 2.0)
    np.fill_diagonal(rho, 1.0)
    for r in range(M - 1):
        rho[r + 1, r] = 0.5
    if cycle:
        rho[0, M - 1] = 0.5
    return rho

def glv_run(G, rho, T, eps=1e-6, eta=0.0, rng=None, slot_of=None, n_slots=None, ior=False):
    """Batch simulation. G: (E, M) with values in [0, 1]; the stop input is u = 1 - G
    (G = 1: no input, the intrinsic sequence continues; G = 0: full stop input). Returns
    first-crossing times (E, M) of x_r > 0.5 (inf if never). With ior=True, a stop input is
    also applied to unit r once the slot it reads (slot_of[e, r]) has been visited by another
    unit (inhibition of return)."""
    E, M = G.shape
    X = np.full((E, M), eps); X[:, 0] = 1.0
    tc = np.full((E, M), np.inf); tc[:, 0] = 0.0
    if ior:
        vis = np.zeros((E, n_slots), dtype=bool)
        vis[np.arange(E), slot_of[:, 0]] = True
        owner = np.full((E, n_slots), -1); owner[np.arange(E), slot_of[:, 0]] = 0
    steps = int(T / DT)
    for k in range(steps):
        Gt = G
        if ior:
            v = vis[np.arange(E)[:, None], slot_of]                       # slot already visited?
            mine = owner[np.arange(E)[:, None], slot_of] == np.arange(M)[None, :]
            Gt = G * (~v | mine)
        grow = 1.0 - (1.0 - Gt) - X @ rho.T          # intrinsic growth 1, stop input u = 1 - G
        X = X + DT * X * grow + DT * eps
        if eta > 0:
            X = X + eta * np.sqrt(np.maximum(X, 0) * DT) * rng.standard_normal(X.shape)
        X = np.maximum(X, 0.0)
        new = (X > 0.5) & np.isinf(tc)
        if new.any():
            tc[new] = (k + 1) * DT
            if ior:
                e_idx, r_idx = np.nonzero(new)
                s_idx = slot_of[e_idx, r_idx]
                free = owner[e_idx, s_idx] < 0
                vis[e_idx[free], s_idx[free]] = True; owner[e_idx[free], s_idx[free]] = r_idx[free]
    return tc

def slots_for(sp, a, M, wrap):
    """Slot read by coordinate unit r from anchor a (or -1 if it does not exist)."""
    out = np.full(M, -1)
    for r in range(M):
        j = a + r
        if sp.kind == "ring": j = j % sp.n
        elif j >= sp.n: continue
        out[r] = j
    return out

# ------------------------------------------------------------------------------------------
# H1: recognition
# ------------------------------------------------------------------------------------------
def recognise_batch(events, M=6, cycle=False, eps=1e-6, eta=0.0, rng=None, T=None):
    """Heteroclinic recognition of a batch of accepted placements. For a cycle, all events
    must share the ring size and M = L. Returns (patterns, info list)."""
    E = len(events)
    G = np.zeros((E, M)); S = np.zeros((E, M), dtype=int); A = []; SAL = []
    for e, ev in enumerate(events):
        sp = ev["space"]
        dplus, dminus, hminus = cs.mismatch(ev["before"], ev["after"], ev["c"])
        sal = ((dplus + dminus).sum(axis=1) > 0).astype(int)
        a = cs.anchor_of(sp, sal, "onset")
        sl = slots_for(sp, a, M, cycle)
        G[e] = [1.0] + [float(sl[r] >= 0 and sal[sl[r]]) for r in range(1, M)]
        S[e] = np.maximum(sl, 0); A.append(a); SAL.append((sal, dplus, dminus, hminus))
    T = T or (M + 2) * (np.log(1 / eps) / LAM + 10)
    tc = glv_run(G, make_rho(M, cycle), T, eps, eta, rng)
    pats, infos = [], []
    for e, ev in enumerate(events):
        sal, dplus, dminus, hminus = SAL[e]
        V = [r for r in range(M) if np.isfinite(tc[e, r])]
        slots = sorted(set(int(S[e, r]) for r in V))
        residual = int(sal.sum() - len(slots))
        role = all(dplus[S[e, r]] @ hminus >= cs.KC - 1 and not dminus[S[e, r]].any() for r in V)
        w = len(cs.SHAPES[ev["space"].kind][ev["shape"]])
        ext = sorted(V) == list(range(w))
        if role and ext and residual == 0:
            p = ("DELTA",)
        else:
            p = (tuple(V), role, ext, residual)
        pats.append(p); infos.append(dict(V=V, tc=tc[e], anchor=A[e]))
    return pats, infos

# ------------------------------------------------------------------------------------------
# H2: execution and guard with the same chain
# ------------------------------------------------------------------------------------------
def execute_exhaustive(kind, envs, widths=(1, 2, 3), M=6, eps=1e-6):
    rng = np.random.default_rng(17)
    cases = []
    for L in envs:
        sp = cs.Space(kind, L)
        for bits in range(2 ** L):
            occ = np.array([(bits >> i) & 1 for i in range(L)])
            ids = occ * int(rng.integers(1, 4))
            for t in range(L):
                for w in widths:
                    cases.append((sp, ids, t, w))
    E = len(cases)
    Gg = np.zeros((E, M)); Gw = np.zeros((E, M)); S = np.zeros((E, M), dtype=int)
    for e, (sp, ids, t, w) in enumerate(cases):
        sl = slots_for(sp, t, M, kind == "ring")
        for r in range(M):
            exists = sl[r] >= 0
            Gw[e, r] = float(r == 0 or (r < w and exists))
            Gg[e, r] = float(r < w and exists and ids[sl[r]] == 0)
        S[e] = np.maximum(sl, 0)
    # the guard: the bump starts only if the target itself is free (unit 0 carries no stop input)
    T = (4 + 2) * (np.log(1 / eps) / LAM + 10)
    tcg = glv_run(np.where(Gg[:, :1] > 0, Gg, 0) + 0.0, make_rho(M), T, eps)
    tcw = glv_run(Gw, make_rho(M), T, eps)
    out = dict(cases=E, permitted=0, false_perm=0, false_ref=0, write_err=0, new_width=0)
    written = []
    for e, (sp, ids, t, w) in enumerate(cases):
        fp = cs.footprint(sp, t, f"w{w}")
        world_ok = fp is not None and not ids[fp].any()
        start_free = Gg[e, 0] > 0
        F = [r for r in range(M) if np.isfinite(tcg[e, r])] if start_free else []
        n_minus = any(r not in F for r in range(w))                     # W_r = 1 but not traced
        permit = not n_minus
        out["false_perm"] += int(permit and not world_ok); out["false_ref"] += int(world_ok and not permit)
        if not permit: continue
        out["permitted"] += 1; out["new_width"] += int(w == 3)
        c = int(cs.TEST_COLOURS[e % 3])
        pred = ids.copy()
        for r in range(M):
            if np.isfinite(tcw[e, r]): pred[S[e, r]] = c
        after = ids.copy(); after[fp] = c
        out["write_err"] += int(not np.array_equal(pred, after))
        written.append(dict(space=sp, before=ids, after=pred, c=c, shape=f"w{w}"))
    return out, written

# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    res = {}
    EV = {k: cs.play(k, 400, np.random.default_rng(3)) for k in ("line", "ring")}

    print("H1. RECOGNITION: heteroclinic chain of coordinate saddles vs breadth-first construction")
    h1 = {}
    for k in ("line", "ring"):
        E = EV[k]
        pats, inf = recognise_batch(E, M=6)
        bfs = [cs.canonical(ev)[0] for ev in E]
        agree = sum((p == ("DELTA",)) == (b[0].startswith("DELTA")) for p, b in zip(pats, bfs))
        row = dict(events=len(E), patterns=len(set(pats)), validated=sum(p == ("DELTA",) for p in pats),
                   agree_with_bfs=agree)
        if k == "ring":                       # the same events with a heteroclinic CYCLE per ring size
            pc = {}
            for L in sorted(set(ev["space"].n for ev in E)):
                sub = [ev for ev in E if ev["space"].n == L]
                p2, _ = recognise_batch(sub, M=L, cycle=True)
                pc[L] = (len(sub), sum(p == ("DELTA",) for p in p2), len(set(p2)))
            row["cycle_by_L"] = {L: dict(events=a, validated=b, patterns=c) for L, (a, b, c) in pc.items()}
        h1[k] = row
        print(f"   {k:5s} {row['events']} events: {row['patterns']} pattern(s); validated {row['validated']}; "
              f"same verdict as breadth-first {agree}/{len(E)}" + (f"; cycle: {row['cycle_by_L']}" if k == "ring" else ""))
    res["H1"] = h1

    print("\nH2. EXECUTION AND GUARD with the same chain (exhaustive states, widths 1-3, colours 4-6)")
    h2 = {}
    for k, envs in (("line", [4, 5, 6, 7, 8]), ("ring", [5, 6, 7, 8, 9])):
        o, written = execute_exhaustive(k, envs)
        rp, _ = recognise_batch(written, M=6)
        o["rerecognised"] = sum(p == ("DELTA",) for p in rp)
        h2[k] = o
        print(f"   {k:5s} cases {o['cases']}; permitted {o['permitted']} (width 3: {o['new_width']}); "
              f"false perm {o['false_perm']}; false ref {o['false_ref']}; write errors {o['write_err']}; "
              f"re-recognised as DELTA {o['rerecognised']}/{o['permitted']}")
    res["H2"] = h2

    print("\nH3. AN EVENT COVERING A WHOLE RING (no onset; fallback anchor; hand width = L)")
    h3 = {}
    for L in range(4, 10):
        sp = cs.Space("ring", L)
        ev = dict(space=sp, before=np.zeros(L, int), after=np.full(L, 2), c=2, shape=None)
        a = cs.anchor_of(sp, np.ones(L, int), "onset")
        M = 12
        sl = slots_for(sp, a, M, True)
        G = np.ones((1, M)); T = (M + 2) * (np.log(1e6) / LAM + 10)
        V_int = [r for r in range(M) if np.isfinite(glv_run(G, make_rho(M), T)[0, r])]
        V_ior = [r for r in range(M) if np.isfinite(glv_run(G, make_rho(M), T, slot_of=sl[None, :],
                                                          n_slots=L, ior=True)[0, r])]
        tcc = glv_run(np.ones((1, L)), make_rho(L, cycle=True), T)[0]
        V_cyc = [r for r in range(L) if np.isfinite(tcc[r])]
        h3[L] = dict(integer=V_int, integer_ior=V_ior, cycle=V_cyc,
                     integer_ok=V_int == list(range(L)), ior_ok=V_ior == list(range(L)), cycle_ok=V_cyc == list(range(L)))
        print(f"   L={L}: integer chain traces {len(V_int)} coordinates (ok={h3[L]['integer_ok']}); "
              f"with inhibition of return {len(V_ior)} (ok={h3[L]['ior_ok']}); cycle {len(V_cyc)} (ok={h3[L]['cycle_ok']})")
    # the cycle does not stop: count laps of a 4-cycle in a fixed window
    Lc, Tw = 4, 400.0
    X = np.full((1, Lc), 1e-6); X[0, 0] = 1; rho = make_rho(Lc, True); entries = 0; cur = 0
    for k in range(int(Tw / DT)):
        X = np.maximum(X + DT * X * (1 - X @ rho.T) + DT * 1e-6, 0)
        lead = int(np.argmax(X[0]))
        if lead != cur and X[0, lead] > 0.5: entries += 1; cur = lead
    h3["cycle_laps_in_400"] = entries / Lc
    print(f"   the 4-cycle keeps running: {entries} saddle changes in t=400 ({entries / Lc:.1f} laps); "
          f"its traced SET stays {{0,1,2,3}}")
    res["H3"] = h3

    print("\nH4. TIMING: dwell per saddle and traversal time")
    h4 = {}
    for eps in (1e-3, 1e-6, 1e-9):
        M = 8; G = np.zeros((7, M))
        for e in range(7): G[e, :e + 2] = 1                       # extents 1..7 open units 0..e
        G[:, 0] = 1
        tc = glv_run(G, make_rho(M), (M + 2) * (np.log(1 / eps) / LAM + 10), eps)
        t_last = [tc[e, e + 1] for e in range(7)]                  # crossing time of the last unit
        slope = np.polyfit(np.arange(1, 8), t_last, 1)[0]
        h4[str(eps)] = dict(dwell=float(slope), predicted=float(np.log(1 / eps) / LAM))
        print(f"   eps={eps:.0e}: dwell per saddle {slope:6.2f}   ln(1/eps)/lambda = {np.log(1 / eps) / LAM:6.2f}")
    xs = [np.log(1 / float(k)) for k in h4]; ys = [h4[k]["dwell"] for k in h4]
    fit = np.polyfit(xs, ys, 1)
    h4["fit_slope_vs_ln_inv_eps"] = float(fit[0]); h4["fit_offset"] = float(fit[1])
    print(f"   fitted dwell = {fit[0]:.3f} ln(1/eps) {fit[1]:+.2f}   (prediction: slope 1/lambda = {1 / LAM:.2f})")
    res["H4"] = h4

    print("\nH5. NOISE SCAN: 100 independent realisations x extents 1..6 per noise level (line chain, M = 8)")
    ETAS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8)
    NREAL, EXT, M8 = 100, range(1, 7), 8
    h5 = {"scan": {}, "training": {}}
    rows = [(e, k) for k in range(NREAL) for e in EXT]
    G = np.zeros((len(rows), M8))
    for i, (e, k) in enumerate(rows): G[i, :e] = 1.0          # the event covers coordinates 0..e-1
    G[:, 0] = 1.0
    for eta in ETAS:
        tc = glv_run(G, make_rho(M8), 1500.0, 1e-6, eta, np.random.default_rng(1000 + int(eta * 1000)))
        stats = dict(correct=0, order_errors=0, premature=0, overrun=0); dw = []
        for i, (e, k) in enumerate(rows):
            V = [r for r in range(M8) if np.isfinite(tc[i, r])]
            t = tc[i, V]
            ordered = bool(np.all(np.diff(t) > 0)) and V == list(range(len(V)))
            stats["order_errors"] += int(not ordered)
            stats["premature"] += int(max(V) < e - 1)
            stats["overrun"] += int(max(V) > e - 1)
            stats["correct"] += int(ordered and V == list(range(e)))
            if ordered and V == list(range(e)) and len(V) > 1: dw += list(np.diff(t))   # correct sequences only
        n = len(rows)
        row = {k2: v / n for k2, v in stats.items()}
        row["dwell_mean"] = float(np.mean(dw)) if dw else None
        row["dwell_cv"] = float(np.std(dw) / np.mean(dw)) if dw else None
        h5["scan"][eta] = row
        print(f"   eta={eta:<5}: P(correct sequence) {row['correct']:.3f}; order errors {row['order_errors']:.3f}; "
              f"premature stops {row['premature']:.3f}; overruns {row['overrun']:.3f}; "
              + (f"dwell {row['dwell_mean']:6.2f} (CV {row['dwell_cv']:.2f})" if dw else "no correct sequence"))
    for k in ("line", "ring"):
        for eta in (0.1, 0.2, 0.4):
            pats, _ = recognise_batch(EV[k], M=6, eta=eta, rng=np.random.default_rng(41))
            h5["training"][f"{k}_{eta}"] = sum(p == ("DELTA",) for p in pats)
        print(f"   {k:5s} training placements validated at eta 0.1 / 0.2 / 0.4: "
              + " / ".join(str(h5['training'][f'{k}_{e}']) for e in (0.1, 0.2, 0.4)) + f"  (of {len(EV[k])})")
    # why training placements fare better at eta = 0.4: extents 1-2 only, two readout windows
    rows2 = [(e, k) for k in range(300) for e in (1, 2)]
    G2 = np.zeros((len(rows2), 6)); G2[:, 0] = 1.0
    for i, (e, k) in enumerate(rows2): G2[i, :e] = 1.0
    h5["extents12_eta0.4"] = {}
    for Tw in ((6 + 2) * (np.log(1e6) / LAM + 10), 1500.0):
        tc = glv_run(G2, make_rho(6), Tw, 1e-6, 0.4, np.random.default_rng(7))
        ok = sum([r for r in range(6) if np.isfinite(tc[i, r])] == list(range(e)) for i, (e, k) in enumerate(rows2))
        h5["extents12_eta0.4"][f"{Tw:.0f}"] = ok / len(rows2)
        print(f"   extents 1-2 only, eta=0.4, window t={Tw:.0f}: P(correct) {ok / len(rows2):.3f}")
    res["H5"] = h5
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = [e if e > 0 else 0.003 for e in ETAS]
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
        ax[0].plot(xs, [h5["scan"][e]["correct"] for e in ETAS], "o-", color="k")
        ax[0].set_xscale("log"); ax[0].set_ylim(-0.03, 1.03); ax[0].set_xlabel(r"noise $\eta$ (0 plotted at 0.003)")
        ax[0].set_ylabel("P(correct sequence)")
        ok = [i for i, e in enumerate(ETAS) if h5["scan"][e]["dwell_cv"] is not None]
        xo = [xs[i] for i in ok]
        ax[1].plot(xo, [h5["scan"][ETAS[i]]["dwell_cv"] for i in ok], "s-", color="tab:red", label="CV of dwell")
        ax[1].set_xscale("log"); ax[1].set_xlabel(r"noise $\eta$"); ax[1].set_ylabel("CV of dwell", color="tab:red")
        ax2 = ax[1].twinx(); ax2.plot(xo, [h5["scan"][ETAS[i]]["dwell_mean"] for i in ok], "^--", color="tab:blue")
        ax[1].set_xlim(ax[0].get_xlim())
        ax2.set_ylabel("mean dwell", color="tab:blue")
        fig.tight_layout(); fig.savefig("fig_noise_scan.pdf"); print("   figure written: fig_noise_scan.pdf")
    except ImportError:
        print("   (matplotlib not available: figure skipped)")

    print("\nH6. STOP STRENGTH: extent 3 on a line chain; stop input u on coordinate 3")
    h6 = {}
    for u in (0.0, 0.25, 0.45, 0.55, 0.75, 1.0):
        G = np.ones((1, 8)); G[0, 3:] = 1.0 - u
        tc = glv_run(G, make_rho(8), 9 * (np.log(1e6) / LAM + 30), 1e-6)[0]
        V = [r for r in range(8) if np.isfinite(tc[r])]
        h6[u] = dict(visited=len(V), stopped_at_extent=V == [0, 1, 2])
        print(f"   u={u:<4}: traced {len(V)} coordinates; stops at the extent: {V == [0, 1, 2]}")
    res["H6"] = h6

    with open("numbers_heteroclinic.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwritten numbers_heteroclinic.json")
