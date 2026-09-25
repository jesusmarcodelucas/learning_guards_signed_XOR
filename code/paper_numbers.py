# Numeros para la nota: varias semillas, errores por anchura en el test exhaustivo, tamano del test.
import numpy as np, importlib.util, json
spec = importlib.util.spec_from_file_location("cp", "cajas_pilot.py"); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
out = {"seeds": []}
for seed in range(5):
    cp.RNG = np.random.default_rng(seed)
    trans, tries = cp.play([4, 5, 6], episodes=400)
    veto, cnt = cp.learn_veto(tries); veto_pp, _ = cp.learn_veto(tries, pooled=False)
    n, fo, fc = cp.audit(tries, veto)
    nE, foE, fcE = cp.exhaustive(veto); nA, foA, fcA = cp.exhaustive(veto_pp)
    out["seeds"].append(dict(seed=seed, trans=len(trans), tries=len(tries), accepted=sum(t[-1] for t in tries),
        Nminus=cnt["N-"], Nplus=cnt["N+"], veto=veto, veto_pp=veto_pp, audit=(n, fo, fc), exh=(nE, foE, fcE), exh_pp=(nA, foA, fcA),
        distractors={u: cnt[u] for u in cp.DISTRACTORES}))
# per-width errors of the ablation, seed 3 (the pilot's default)
cp.RNG = np.random.default_rng(3); trans, tries = cp.play([4, 5, 6], episodes=400)
veto, _ = cp.learn_veto(tries); veto_pp, _ = cp.learn_veto(tries, pooled=False)
per_w = {}
for w in range(1, 7):
    nE, foE, fcE = cp.exhaustive(veto, widths=(w,)); nA, foA, fcA = cp.exhaustive(veto_pp, widths=(w,))
    per_w[w] = dict(cases=nE, pooled=(foE, fcE), local=(foA, fcA))
out["per_width"] = per_w
# counts of impossible/possible cases in the exhaustive set
imp = pos = 0
for L in (4, 5, 6, 7, 8):
    for bits in range(2 ** L):
        cells = np.array([(bits >> i) & 1 for i in range(L)], dtype=int)
        for p in range(0, L + 2):
            for w in range(1, 7):
                ok, _ = cp.attempt(cells, p, w); pos += ok; imp += not ok
out["exhaustive_possible"] = pos; out["exhaustive_impossible"] = imp
json.dump(out, open("paper_numbers.json", "w"), indent=1)
for s in out["seeds"]: print(s["seed"], s["tries"], s["Nminus"], s["Nplus"], s["veto"], s["veto_pp"], s["audit"], s["exh"], s["exh_pp"])
print(per_w); print("possible/impossible", pos, imp)
