#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circuit_pilot.py -- Canonicalizacion como CIRCUITO, sin operaciones simbolicas select/relocate/substitute/bind.

Tres primitivos y una conjuncion, todos operaciones de vectores:
  (M) mismatch con signo por slot y unidad de codigo: d = s' - s -> canales d+ y d-.
  (P) propagacion con compuerta desde un ancla: un 'bump' en un atractor de linea de K posiciones se
      resetea en el ancla y avanza (D+) mientras el siguiente slot cumpla una condicion local.
      Con la condicion 'hay mismatch' recorre el cambio y cuenta su extension; con la condicion
      'esta libre' produce H, el termometro del hueco (la misma cadena que usa la guarda).
  (C) conjuncion papel x filler: el papel X se activa por coincidencia entre el codigo que
      desaparece de la mano (canal d-) y el codigo que aparece en los slots recorridos (canal d+).
El patron canonico Delta se lee de poblaciones de COORDENADA RELATIVA (R_0, R_1, ...) y del papel X;
nunca de indices absolutos ni de nombres de color. Se mide cuantos engramas distintos produce, y que
pasa al quitar cada pieza del circuito.
"""
import numpy as np, importlib.util
from collections import Counter
spec = importlib.util.spec_from_file_location("cp", "cajas_pilot.py"); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
spec = importlib.util.spec_from_file_location("op", "operator_pilot.py"); op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)

N, KC, K, NCOL = 64, 6, 6, 6                       # codigo de filler: KC unos entre N; K = rango del atractor
RNGc = np.random.default_rng(5)
CODE = np.zeros((NCOL + 1, N), dtype=int)           # fila 0 = vacia (todo ceros); filas 1..6 = colores
for c in range(1, NCOL + 1): CODE[c, RNGc.choice(N, KC, replace=False)] = 1
PROJ = RNGc.standard_normal((512, 3 * K + 2))       # expansion dispersa fija para el engrama

def thermo(x):  v = np.zeros(K, dtype=int); v[:min(x, K)] = 1; return v

# ------------------------------------------------------------------ (M) mismatch
def mismatch(before, after, hand_c):
    """d+ y d- por casilla (matriz L x N) y d- de la mano (vector N: el color que la deja)."""
    S0 = CODE[before]; S1 = CODE[after]; d = S1 - S0
    dplus, dminus = (d > 0).astype(int), (d < 0).astype(int)
    hand_minus = CODE[hand_c + 1]                   # la mano pasa de 'color' a nada: su canal d- es el codigo del color
    return dplus, dminus, hand_minus

# ------------------------------------------------------------------ (P) propagacion con compuerta
def propagate(cond, anchor, L):
    """Bump reseteado en 'anchor'; avanza mientras cond[anchor + r] sea 1 y r < K.
    Devuelve la lista de compuertas (one-hot sobre slots) visitadas y su numero (la extension)."""
    gates = []
    for r in range(K):
        j = anchor + r
        if j >= L or not cond[j]: break
        g = np.zeros(L, dtype=int); g[j] = 1; gates.append(g)
    return gates, len(gates)

# ------------------------------------------------------------------ el circuito completo
def canonical(before, after, hand_c, hand_w, reset=True, bind_role=True, bind_extent=True):
    dplus, dminus, hminus = mismatch(before, after, hand_c)
    L = len(before)
    salience = ((dplus + dminus).sum(axis=1) > 0).astype(int)          # OR sobre unidades
    anchor = int(np.argmax(salience * np.arange(L, 0, -1)))             # WTA con sesgo de onset (izquierda)
    if not salience.any(): return None
    gates, extent = propagate(salience, anchor, L)                       # recorrido del cambio
    R = np.stack([g @ dplus for g in gates] + [np.zeros(N, dtype=int)] * (K - extent))   # R_r: codigo d+ bajo el bump
    # (C) conjuncion: X se activa en R_r si el codigo que aparece ahi coincide con el que dejo la mano
    Xr = ((R @ hminus) >= KC - 1).astype(int) if bind_role else None
    residual = int(salience.sum() - extent)                               # mismatch fuera del recorrido
    # comparacion de extensiones con el mismo comparador: termometro recorrido vs termometro de la mano
    ext_match = int(not (thermo(extent) - thermo(hand_w)).any())
    if reset:
        pos = np.zeros(K, dtype=int); pos[:extent] = 1                    # coordenadas relativas ocupadas
    else:
        pos = np.zeros(K, dtype=int); pos[:] = 0                          # sin reset: coordenadas ABSOLUTAS
        absol = np.zeros(8, dtype=int); absol[anchor] = 1
    if bind_role:
        role = Xr[:extent].all() and Xr[extent:].sum() == 0               # X ocupa exactamente el recorrido
        who = np.array([1], dtype=int)                                    # 'X'
    else:
        role = True; who = CODE[hand_c + 1]                               # sin sustitucion: el color pasa al patron
    if bind_extent and ext_match and role and residual == 0:
        pattern = np.concatenate([np.array([1]), who, np.array([1])])     # X llena el recorrido; X salio de la mano
    else:
        pattern = np.concatenate([pos, who, np.array([int(role)]), np.array([ext_match, residual])])
    if not reset: pattern = np.concatenate([pattern, absol])
    return tuple(pattern.tolist()), dict(anchor=anchor, extent=extent, ext_match=ext_match, role=bool(role), residual=residual)

def engram(pattern):
    v = np.zeros(3 * K + 2); v[:len(pattern)] = pattern[:3 * K + 2]
    return frozenset(np.argsort(PROJ @ v)[-16:].tolist())

# ------------------------------------------------------------------ aplicacion de Delta (ligar y escribir por el mismo enrutado)
def propagate_gate(gate_seq, anchor, L):
    """P con compuerta TERMOMETRICA: el bump avanza mientras la unidad r del termometro este activa.
    Es la misma dinamica que propagate(); solo cambia de donde viene la compuerta."""
    gates = []
    for r in range(K):
        j = anchor + r
        if j >= L or not gate_seq[r]: break
        g = np.zeros(L, dtype=int); g[j] = 1; gates.append(g)
    return gates, len(gates)

def apply_delta(cells_id, p, hand_c, hand_w):
    """Escribe Delta con el mismo circuito al reves: reset del bump en p, X <- color de la mano, y el bump
    avanza mientras W_r (termometro de la anchura de la mano) este activo, escribiendo X (x) color."""
    pred = cells_id.copy(); L = len(pred)
    gates, ext = propagate_gate(thermo(hand_w), p - 1, L)          # P(W): compuerta = termometro de la mano
    for g in gates:
        pred[np.argmax(g)] = hand_c + 1
    return pred

if __name__ == "__main__":
    cp.RNG = np.random.default_rng(3); trans, tries = cp.play([4, 5, 6], episodes=400)
    T = op.colour_transitions(trans, op.NCOL_TRAIN)
    print(f"{len(T)} colocaciones, {op.NCOL_TRAIN} colores; codigos de {KC} unos entre {N}; atractor de K = {K} posiciones\n")
    print("CANONICALIZACION COMO CIRCUITO: engramas distintos en las 1486 colocaciones")
    full = [canonical(tr["before"], tr["after"], tr["c"], tr["w"]) for tr in T]
    ok = sum(m["ext_match"] and m["role"] and m["residual"] == 0 for _, m in full)
    print(f"  circuito completo (mismatch + propagacion con reset + conjuncion X + comparacion de extensiones): "
          f"{len(set(engram(p) for p, _ in full))} engrama(s); esquema validado en {ok}/{len(T)}")
    for name, kw in [("sin reset en el ancla (coordenadas absolutas)", dict(reset=False)),
                     ("sin conjuncion X (el color pasa al patron)", dict(bind_role=False)),
                     ("sin comparacion de extensiones (w queda en el patron)", dict(bind_extent=False))]:
        pats = [canonical(tr["before"], tr["after"], tr["c"], tr["w"], **kw)[0] for tr in T]
        print(f"  ablacion, {name:<58s} {len(set(pats)):3d} patrones distintos")
    print(f"  (referencia: estado crudo antes/despues = {len(set(op.key_raw(tr) for tr in T))} eventos)")
    ext = Counter(m["extent"] for _, m in full); anc = Counter(m["anchor"] for _, m in full)
    print(f"  extension recorrida (canal aparte): {dict(sorted(ext.items()))};  anclas: {dict(sorted(anc.items()))}")

    print("\nH POR EL MISMO PRIMITIVO: propagar desde el ancla mientras la casilla este libre")
    c = np.array([1, 1, 0, 1, 1, 0]); g, h = propagate(1 - c, 2, 6); print(f"  casillas {c.tolist()}, ancla 3 -> hueco {h}, H = {thermo(h).tolist()}")

    # reuso: predecir con Delta ligada a colores nuevos, sobre el test exhaustivo, solo donde G lo permite
    veto, _ = cp.learn_veto(tries); n = bad = 0
    for L in (4, 5, 6, 7, 8):
        for bits in range(2 ** L):
            cells = np.array([(bits >> i) & 1 for i in range(L)], dtype=int)
            for p in range(1, L + 1):
                for w in range(1, 7):
                    u = cp.units(cells, p, cp.perceive_box(w))
                    if any(u[v] for v in veto): continue
                    okw, nc = cp.attempt(cells, p, w); c = int(RNGc.integers(3)) + 3   # colores 3-5, nunca vistos
                    ids = cells * 9; world = ids.copy(); world[p - 1:p - 1 + w] = c + 1
                    pred = apply_delta(ids, p, c, w); n += 1; bad += int(not okw or not np.array_equal(pred, world))
    print(f"\nREUSO: Delta ligada a colores nuevos y escrita por el mismo enrutado: {n} colocaciones permitidas por G, errores {bad}")
