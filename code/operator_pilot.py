#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
operator_pilot.py -- La cadena completa sobre el mundo de cajas, con FILLERS con identidad.

  slots/fillers -> canonicalizacion (select, relocate, substitute, bind) -> diferencia con signo
  -> cambio reutilizable Delta -> operador O = (G, Delta), con G aprendida (cajas_pilot.learn_veto).

Estado s(t): slots persistentes = las L casillas y la mano. Filler de una casilla = vacia o la
IDENTIDAD de la caja que la ocupa (un color entre varios); filler de la mano = (color, anchura) o nada.
Cada filler es un codigo binario; aqui, un bit por color (one-hot) y dos bits de anchura.
La canonicalizacion se ASUME como arquitectura fija (no se aprende), y se mide que aporta cada paso.
Los numeros de la guarda no cambian: los colores salen de un generador aparte.
"""
import numpy as np, importlib.util
from collections import Counter
spec = importlib.util.spec_from_file_location("cp", "cajas_pilot.py"); cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
CRNG = np.random.default_rng(11)                       # generador APARTE para los colores
NCOL_TRAIN, NCOL_TEST = 3, 3                           # colores 0-2 en entrenamiento, 3-5 nunca vistos

def colour_transitions(trans, ncol, offset=0):
    """A cada colocacion aceptada (s, s', p, w) se le asigna un color de caja. Devuelve estados con
    identidad: cells_id (0 = vacia, c+1 = caja de color c) y mano (color, w) antes / nada despues."""
    out = []
    for cells, nc, p, w in trans:
        c = int(CRNG.integers(ncol)) + offset
        # reconstruir identidades de las casillas ya ocupadas: color al azar por bloque (no importa para d)
        ids = np.zeros(len(cells), dtype=int)
        k = 0
        while k < len(cells):
            if cells[k] == 1:
                j = k
                while j < len(cells) and cells[j] == 1: j += 1
                ids[k:j] = int(CRNG.integers(ncol)) + offset + 1; k = j
            else: k += 1
        ids2 = ids.copy(); ids2[p - 1:p - 1 + w] = c + 1
        out.append(dict(before=ids, after=ids2, hand_before=(c, w), hand_after=None, p=p, w=w, c=c))
    return out

def changed_slots(tr):
    """SELECT, alimentado por el comparador: d = s' - s por slot; se quedan los slots con d != 0.
    Aqui la identidad se codifica one-hot por color, asi que d != 0 en una casilla equivale a que su
    filler cambio. Devuelve [(indice, filler_antes, filler_despues)] mas el slot de la mano."""
    d = [(i, int(tr["before"][i]), int(tr["after"][i])) for i in range(len(tr["before"])) if tr["before"][i] != tr["after"][i]]
    hand = ("hand", tr["hand_before"], tr["hand_after"])
    return d, hand

def key_raw(tr):        # estado entero antes y despues, en slots absolutos: nada canonicalizado
    return (tuple(tr["before"]), tr["hand_before"], tuple(tr["after"]), tr["hand_after"])
def key_signed(tr):     # SELECT: solo los slots con d != 0, con identidad y posicion absoluta
    d, hand = changed_slots(tr)
    return (len(tr["before"]), tuple((i + 1, b, a) for i, b, a in d), hand)
def key_relocate(tr):   # RELOCATE: posiciones relativas al ancla (el primer slot con d != 0)
    d, hand = changed_slots(tr); a0 = d[0][0]
    return (tuple((i - a0, b, a) for i, b, a in d), hand)
def key_substitute(tr): # SUBSTITUTE: la identidad de la caja que se mueve pasa a ser el papel X
    d, hand = changed_slots(tr); a0 = d[0][0]; c = tr["hand_before"][0] + 1   # el color que estaba en la mano
    sub = lambda f: "vacia" if f == 0 else ("X" if f == c else f"otro:{f}")
    return (tuple((i - a0, sub(b), sub(a)) for i, b, a in d), ("hand", ("X", tr["hand_before"][1]), hand[2]))
def key_bind(tr):
    """BIND: deriva y VALIDA el esquema parametrizado. Solo devuelve Delta(X, p, w) si se cumple todo:
       (i) los slots cambiados son consecutivos desde el ancla; (ii) su numero es la anchura w que
       la mano declaraba; (iii) todos pasan de 'vacia' a X, el mismo X que desaparece de la mano;
       (iv) la mano queda vacia. Si algo falla, el evento NO se identifica con el esquema."""
    d, hand = changed_slots(tr); a0 = d[0][0]; c = tr["hand_before"][0] + 1; w = tr["hand_before"][1]
    consecutive = [i - a0 for i, b, a in d] == list(range(len(d)))
    cardinality = len(d) == w
    same_X = all(b == 0 and a == c for i, b, a in d)
    hand_empty = tr["hand_after"] is None
    if consecutive and cardinality and same_X and hand_empty:
        return ("Delta", "[p, p+w-1] <- X", "hand <- nada")
    return ("INVALID", tuple((i - a0, b, a) for i, b, a in d), hand)

if __name__ == "__main__":
    cp.RNG = np.random.default_rng(3)
    trans, tries = cp.play([4, 5, 6], episodes=400)
    T = colour_transitions(trans, NCOL_TRAIN)
    print(f"{len(T)} colocaciones aceptadas, {NCOL_TRAIN} colores de caja, estanterias de 4, 5 y 6\n")
    print("ESCALERA: cuantos eventos distintos hay a cada paso de la canonicalizacion (arquitectura fija)")
    for name, key in [("estado crudo antes/despues, en slots absolutos", key_raw),
                      ("SELECT (por el comparador: slots con d != 0), identidad y posicion absoluta", key_signed),
                      ("RELOCATE: posiciones relativas al ancla", key_relocate),
                      ("SUBSTITUTE: la caja concreta pasa a ser el papel X", key_substitute),
                      ("BIND: la anchura es una variable -> un solo cambio canonico Delta", key_bind)]:
        n = len(set(key(tr) for tr in T)); print(f"  {name:<74s} {n:5d}")
    nval = sum(key_bind(tr)[0] == "Delta" for tr in T)
    print(f"  BIND valida el esquema Delta(X, p, w) en {nval}/{len(T)} colocaciones (consecutivas desde el ancla, |cambio| = w, X de la mano a las casillas, mano vacia)")

    # --- Operador O = (G, Delta): Delta canonico + G aprendida de los 'no' (la guarda de cajas_pilot)
    veto, _ = cp.learn_veto(tries)
    print(f"\nOPERADOR O = (G, Delta). G aprendida: 'ninguna unidad de veto activa', vetos = {veto}")
    print("  Delta(X, p, w): casillas p..p+w-1 <- X; mano <- nada.  (X: la caja en la mano, cualquiera)")
    # --- Reuso: aplicar O a fillers nunca vistos (colores 3-5), posiciones y anchuras nunca vistas
    n_ok = n_pred = n_bad = 0
    for L in (4, 5, 6, 7, 8):
        for bits in range(2 ** L):
            cells = np.array([(bits >> i) & 1 for i in range(L)], dtype=int)
            for p in range(0, L + 2):
                for w in range(1, 7):
                    ok, nc = cp.attempt(cells, p, w)
                    u = cp.units(cells, p, cp.perceive_box(w)); belief = not any(u[v] for v in veto)
                    if not belief: continue                     # G dice que no: no se intenta
                    c = int(CRNG.integers(NCOL_TEST)) + NCOL_TRAIN  # color nunca visto
                    ids = cells * 9; ids2 = ids.copy(); ids2[p - 1:p - 1 + w] = c + 1   # prediccion por Delta
                    world = cells * 9; world[p - 1:p - 1 + w] = c + 1 if ok else world[p - 1:p - 1 + w]
                    n_pred += 1; n_ok += ok; n_bad += int(not np.array_equal(ids2, world))
    print(f"\nREUSO del operador con fillers nuevos (colores 3-5), estanterias de 4-8 y anchuras 1-6:")
    print(f"  {n_pred} colocaciones que G permite; el mundo acepta {n_ok}; errores de prediccion de Delta: {n_bad}")
