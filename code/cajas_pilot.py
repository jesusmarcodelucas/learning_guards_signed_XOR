#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cajas_pilot.py -- VERSION DEPOSITADA CON LA NOTA I: todas las unidades del comparador (C+-_k, N+-) y los
distractores tienen conexion plastica al veto; el criterio selecciona N-, C-_1 y C-_2 (Tabla 2 de la nota).
La ablacion sin pooling conserva C-_1 y C-_2 y da 2541 falsos abiertos.

cajas_pilot.py -- El mundo mas pequeno en el que se ven las dos cosas que separa ENGRAMMER-X:

  NIVEL 1, la REGLA:      de los cambios observados y de los "no" de la estanteria se aprende
                          un operador O = (G, Delta, Theta) escrito con variables (p, w),
                          desligando (canonicalizando) el cambio observado.
  NIVEL 2, la ESTRATEGIA: de los atascos (no de los "no") se aprende que conviene hacer para
                          alcanzar un objetivo. Otra senal de error, otro nivel.

MUNDO
  Una estanteria de L casillas en fila. Cajas de anchura w in {1, 2}, altura 1.
  Accion: poner la caja que se tiene en la mano en la casilla p (1 <= p <= L).
  El mundo acepta si las casillas p .. p+w-1 existen y estan vacias; entonces se llenan.
  Si no, dice "no" y NO cambia nada. Cada intento, aceptado o rechazado, es una observacion.

NOTACION (la misma que en el articulo y en la charla)
  s(t)      estado: vector de 0/1, una componente por casilla (0 = vacia, 1 = llena).
            Cada casilla es un SLOT (una posicion fija del vector); su contenido es el FILLER.
            Aqui el filler se codifica con 1 bit; en el sistema real, con 12 unos entre 256.
  d(t)      = s(t+1) - s(t), componente a componente, en {-1, 0, +1}. El COMPARADOR con signo.
  p         ancla: la primera casilla en la que d != 0 (donde ocurre el cambio).
  w         anchura de la caja: numero de +1 en d.
  d_barra   el cambio visto desde el ancla: d_barra[k] = d[p + k].
  Delta(w)  el patron "w unos seguidos". Restado de d_barra deja residuo 0 para toda colocacion.
  e         engrama: la direccion (codigo disperso) a la que van a parar todas las colocaciones.
  G         guarda: condicion, aprendida de los "no", para que Delta ocurra.
  Theta     evidencia: contadores (exitos, fallos), leidos como una distribucion Beta.
  epsilon   = s(t+1) - s_pred(t+1): residuo de la prediccion. 0 si la caja entro; != 0 si no.

Todo es contable a mano. Solo NumPy. Semilla fija para que los numeros se repitan.
"""
import numpy as np
from collections import Counter, defaultdict
RNG = np.random.default_rng(3)

# =============================================================================================
# 1. EL MUNDO
# =============================================================================================
def attempt(cells, p, w):
    """Un intento de colocacion. Devuelve (ok, cells_despues).

    cells : array de 0/1 de longitud L, el estado s(t) de las casillas.
    p     : casilla de destino (1..L). Puede llegar fuera de rango: entonces es un "no".
    w     : anchura de la caja (1 o 2).

    El mundo es el arbitro. No explica nada: solo acepta (y devuelve el estado cambiado)
    o rechaza (y devuelve el MISMO estado, sin cambiar nada). El aprendiz nunca ve estas
    lineas; solo ve s(t), s(t+1) y si el intento fue aceptado.
    """
    L = len(cells)
    if p < 1 or p + w - 1 > L:                 # la caja sobresale de la estanteria
        return False, cells
    if cells[p - 1:p - 1 + w].any():           # alguna casilla de destino ya esta llena
        return False, cells
    nc = cells.copy(); nc[p - 1:p - 1 + w] = 1 # las w casillas desde p pasan de 0 a 1
    return True, nc

def play(lengths, episodes, attempts_per_step=4):
    """Genera la experiencia: episodios de "un nino colocando cajas al azar".

    En cada episodio se elige una estanteria de longitud L (de la lista `lengths`) y se
    empieza vacia. Llegan cajas de anchura aleatoria. Con cada caja:
      - se prueban `attempts_per_step` casillas AL AZAR, incluidas casillas fuera de la
        estanteria (p = 0 o p = L+1). Muchos de estos intentos son rechazados. Son la
        senal con la que se aprende la guarda G (punto 8): sin "no" no hay guarda.
      - despues se coloca la caja en una casilla legal (elegida al azar entre las legales),
        y esa transicion aceptada (s, s') es la que ensena el cambio Delta (puntos 3-6).
    El episodio termina cuando no cabe ninguna caja (ni de 2 ni de 1).

    Devuelve:
      trans : lista de (s, s', p, w) con las colocaciones ACEPTADAS  -> para la escalera.
      tries : lista de (s, p, w, ok) con TODOS los intentos          -> para la guarda.
    """
    trans, tries = [], []
    for _ in range(episodes):
        L = int(RNG.choice(lengths)); cells = np.zeros(L, dtype=int)
        while True:
            w = int(RNG.integers(1, 3))                       # llega una caja de 1 o de 2
            legal = [p for p in range(1, L + 1) if attempt(cells, p, w)[0]]
            if not legal:
                # si la de 2 no cabe pero una de 1 si, cambiamos de caja; si nada cabe, fin
                if w == 2 and any(attempt(cells, p, 1)[0] for p in range(1, L + 1)):
                    w = 1; legal = [p for p in range(1, L + 1) if attempt(cells, p, 1)[0]]
                else:
                    break
            for _ in range(attempts_per_step):                # intentos a ciegas (muchos "no")
                p = int(RNG.integers(0, L + 2))
                ok, _ = attempt(cells, p, w); tries.append((cells.copy(), p, w, ok))
            p = int(RNG.choice(legal)); ok, nc = attempt(cells, p, w)   # y una colocacion real
            tries.append((cells.copy(), p, w, True)); trans.append((cells.copy(), nc.copy(), p, w))
            cells = nc
    return trans, tries

# =============================================================================================
# 2. EL COMPARADOR Y LA CANONICALIZACION (puntos 3, 4 y 5)
# =============================================================================================
def signed_diff(a, b):
    """d = b - a, componente a componente. Vale +1 donde algo aparece, -1 donde algo
    desaparece, 0 donde nada cambia. En el cerebro: dos canales (uno para +1, otro para -1),
    cada uno un relevo excitador con veto inhibidor (el motivo XOR con signo)."""
    return b - a

def canon(before, after, level):
    """El cambio observado, expresado en uno de tres niveles de canonicalizacion.
    Devuelve una CLAVE (tupla) tal que dos colocaciones reciben la misma clave si y solo si
    son "el mismo cambio" a ese nivel. Contar claves distintas = contar reglas distintas.

    level 0 : d tal cual, junto con L. Direcciones ABSOLUTAS: "en la estanteria de 6 se
              llenan las casillas 4 y 5". Cada (L, p, w) es un cambio distinto -> 27.
    level 1 : d visto desde el ancla p (la primera casilla que cambia). Se ha DESLIGADO EL
              DONDE: ya no importa p. Quedan dos patrones: [+1] y [+1, +1]      -> 2.
    level 2 : ademas se lee w (numero de +1) y se resta Delta(w) = [+1]*w. El residuo es 0
              para todas: se ha DESLIGADO LA ANCHURA. Queda una sola regla, con variables
              p y w: "w casillas desde p se llenan"                            -> 1.
    """
    d = signed_diff(before, after); L = len(d)
    if level == 0:
        return (L, tuple(d.tolist()))
    idx = np.flatnonzero(d)                    # casillas donde d != 0
    a = int(idx[0])                            # el ancla p (indice 0-based)
    rel = tuple(d[a:idx[-1] + 1].tolist())     # d_barra: el cambio desde el ancla
    if level == 1:
        return rel
    w = len(rel)                               # la anchura, leida del propio cambio
    residual = [int(x) - 1 for x in rel]       # d_barra - Delta(w): todo ceros
    return ("w casillas desde el ancla se llenan", all(r == 0 for r in residual))

# ---------------------------------------------------------------------------------------------
# 2b. DE DONDE SALE canon(): NO ES UNA ELECCION, ES UNA ARQUITECTURA FIJA
# ---------------------------------------------------------------------------------------------
# Canonicalizar no se aprende ni se escoge: es un circuito fijo que aplica SIEMPRE las mismas
# invariancias, igual que una red convolucional aplica siempre la invariancia a traslacion.
# Aqui, tres, y las tres tienen un modelo neuronal establecido:
#   (1) DONDE : la atencion va al cambio y el patron se lee en coordenadas centradas en ella
#               (remapeo sacadico; campos de ganancia).
#   (2) CUANTO: la ventana atendida se reescala a su extension; la extension sale por un canal
#               aparte (normalizacion, como la retina divide por la luminancia y deja la forma;
#               enrutamiento dinamico de Olshausen, Anderson y Van Essen 1993).
#   (3) QUIEN : la identidad del participante va por otro canal (via ventral), separada del
#               patron del cambio (via dorsal). En la estanteria no hay identidades: se omite.
# El engrama es una funcion fija del patron normalizado: misma entrada, misma direccion.
# El criterio "menos engramas, menos residuo" NO lo calcula el animal: es el criterio de
# DISENO que explica por que existen estas invariancias y no otras (como los filtros de V1).
W = 6   # tamano fijo de la ventana atendida (en unidades de la rejilla)

def canon_fixed(before, after):
    """La operacion fija. Devuelve (patron_normalizado, canales) con
       patron_normalizado : el cambio, centrado en la atencion y reescalado a la ventana W
       canales            : p (donde fue la atencion) y w (extension del cambio)
    No hay ningun parametro que dependa del mundo ni ninguna eleccion."""
    d = signed_diff(before, after); idx = np.flatnonzero(d)
    p = int(idx[0]) + 1                         # (1) la atencion va al cambio
    core = d[idx[0]:idx[-1] + 1]                #     y el patron se lee desde ahi
    w = len(core)                               # (2) la extension, por su canal
    pattern = core[(np.arange(W) * w) // W]     #     y el patron reescalado a la ventana fija
    return tuple(int(x) for x in pattern), (p, w)

def canon_ablation(before, after, sin):
    """Que se rompe si al circuito le falta una invariancia (ablacion de arquitectura).
    sin = 'nada' -> el circuito completo; 'escala' -> sin (2); 'atencion' -> sin (1) ni (2);
    'signo' -> sin el signo del comparador."""
    d = signed_diff(before, after); idx = np.flatnonzero(d)
    if sin == "atencion": return (len(d), tuple(d.tolist()))          # direcciones absolutas
    core = d[idx[0]:idx[-1] + 1]
    if sin == "escala":   return tuple(core.tolist())                   # centrado, sin reescalar
    if sin == "signo":    core = np.abs(core)                           # |d|: no distingue poner de quitar
    return tuple(int(x) for x in core[(np.arange(W) * len(core)) // W])

# =============================================================================================
# 3. LA GUARDA G: NO SE LE DA LA RELACION, SE LE DA UN COMPARADOR Y DOS CANTIDADES (punto 8)
# =============================================================================================
# Antes el codigo calculaba 'hueco >= anchura' como un entero y un booleano y se lo pasaba al
# aprendiz: la relacion estaba regalada. Ahora el aprendiz recibe solo:
#   H : cuanto espacio libre seguido hay desde la casilla atendida, en codigo termometro,
#       GENERADO POR UNA CADENA DE GATING LOCAL (no por un contador simbolico);
#   W : cuanto espacio exige la caja, en el MISMO codigo (es el canal 'w' del operador,
#       ligado a la caja que se tiene en la mano);
#   r = H - W : el comparador con signo, que da dos canales de unidades, C+ y C-.
# La condicion 'hueco >= w' no existe como simbolo. Es, literalmente, "no hay actividad en C-".
# Y ni siquiera eso se escribe: la asociacion C- -> veto del operador SE APRENDE de los "no"
# (inhibicion condicionada: una unidad que esta activa cuando el mundo dice "no" gana peso
# inhibidor sobre el engrama). Los distractores siguen ahi para comprobar que el veto se
# aprende especificamente del canal negativo.

def avail_code(cells, p):
    """H = (h_1..h_W), termometro de la extension libre desde p, por gating local:
         h_1 = e_1,   h_k = h_{k-1} * e_k,   con e_j = 1 si la casilla p+j-1 esta vacia (0 si esta
       llena o fuera de la estanteria). La senal 'hay sitio' se propaga hacia la derecha y se
       detiene en la primera casilla ocupada. Ejemplo: vacias 1,1,0,1,1 -> H = 1,1,0,0,0."""
    L = len(cells); H = np.zeros(W, dtype=int); h = 1
    for k in range(W):
        j = p + k                                          # casilla p+k (1-based)
        e = 1 if (1 <= j <= L and cells[j - 1] == 0) else 0
        h = h * e; H[k] = h
    return H

def perceive_box(w_world):
    """W_agent: la representacion que el agente tiene de la caja que lleva en la mano, en el MISMO
    codigo termometro que H. Es percepcion suministrada (el simulador la construye a partir de la
    anchura fisica w_world, que el aprendiz NUNCA recibe como numero), igual que en el articulo la
    percepcion factorizada se suministra. En un intento fallido d = 0, asi que la extension exigida
    no puede leerse del cambio: viene de aqui, de la caja en la mano."""
    Wc = np.zeros(W, dtype=int); Wc[:w_world] = 1; return Wc

def comparator_units(cells, p, box):
    """r = H - W_agent. Devuelve las unidades binarias que ve el aprendiz:
       C+_k = 1 si r_k = +1 (sobra sitio en k),  C-_k = 1 si r_k = -1 (falta sitio en k),
       y las dos poblaciones convergentes del comparador:
       N+ = OR_k C+_k  (la primera representacion contiene algo que la segunda no),
       N- = OR_k C-_k  (la segunda contiene algo que la primera no).
    La convergencia de un canal en una poblacion es una propiedad GENERICA del comparador, no una
    regla sobre cajas: N- significa siempre 'falta algo', sea cual sea la posicion. Es el mismo
    comparador que da d(t) y epsilon(t), aplicado a dos cantidades."""
    r = avail_code(cells, p) - box
    u = {}
    for k in range(W):
        u[f"C+_{k+1}"] = int(r[k] == 1); u[f"C-_{k+1}"] = int(r[k] == -1)
    u["N+"] = int((r == 1).any()); u["N-"] = int((r == -1).any())
    return u
    # PROPOSICION (la desigualdad no esta programada: resulta de la geometria de los codigos).
    #   Con H_k = 1[k <= h] y W_k = 1[k <= w], r_k = H_k - W_k vale -1 sii k <= w y k > h.
    #   Existe tal k sii w > h. Luego  N- = 1 <=> w > h,  y  N- = 0 <=> h >= w.
    #   Simetricamente N+ = 1 <=> h > w, y N+ = N- = 0 <=> h = w. Vale para cualquier h, w <= W:
    #   el rango representacional del circuito es la unica hipotesis.

DISTRACTORES = ["p_es_par", "la_caja_es_ancha", "casilla_izquierda_llena", "estanteria_vacia", "L_es_par"]
def distractor_units(cells, p, box):
    """Unidades que no tienen nada que ver con que la caja quepa. Estan para comprobar que el
    veto no se aprende de cualquier cosa que coincida con fallos."""
    L = len(cells)
    return {"p_es_par": int(p % 2 == 0), "la_caja_es_ancha": int(box.sum() == 2),
            "casilla_izquierda_llena": int(2 <= p <= L + 1 and cells[p - 2] == 1),
            "estanteria_vacia": int(not cells.any()), "L_es_par": int(L % 2 == 0)}

def units(cells, p, box):
    """Todo lo que ve el aprendiz en un intento: 12 unidades del comparador, sus dos poblaciones
    convergentes N+ y N-, y 5 distractores. Nunca el entero w."""
    u = comparator_units(cells, p, box); u.update(distractor_units(cells, p, box)); return u

def beta_tail(s, f, x=0.9, grid=4000):
    """P( p_fallo > x ) bajo la posterior Beta(1 + f, 1 + s), con prior uniforme.
    Con s = 0, f = 5: 1 - 0.9^6 = 0.47 (no concluyente); con f = 30: 0.96 (si). Por eso se
    exige n >= 5 Y > 0.95: es un umbral de evidencia, no de tasa. Integracion numerica."""
    a, b = 1 + f, 1 + s
    t = np.linspace(1e-6, 1 - 1e-6, grid)
    lp = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t)
    wgt = np.exp(lp - lp.max()); wgt /= wgt.sum()
    return float(wgt[t > x].sum())

PLASTIC_POOLED = ["N+", "N-"] + [f"C{s}_{k+1}" for k in range(W) for s in "+-"] + DISTRACTORES   # nota I: todas las unidades del comparador proyectan al veto
PLASTIC_LOCAL = [f"C{s}_{k+1}" for k in range(W) for s in "+-"] + DISTRACTORES  # ablacion sin convergencia

def learn_veto(tries, pooled=True):
    """Aprende que unidades vetan al operador. Devuelve (veto, cnt).

    ARQUITECTURA PRINCIPAL (pooled=True):   C-_1..C-_W -> N- --(plastica)--> V_O
      Las unidades por posicion C-_k CONVERGEN en N- y no tienen proyeccion propia al veto.
      Solo N+, N- y los distractores tienen sinapsis plasticas hacia el veto del operador.
    ABLACION (pooled=False):                C-_k --(plastica)--> V_O, sin poblacion convergente
      Cada C-_k tiene que aprender por su cuenta; lo que no se ha visto en el entrenamiento
      (anchuras 3-6) no puede vetar. La diferencia entre las dos mide para que sirve el pooling.

    REGLA LOCAL (inhibicion condicionada, tres factores)
      Para cada unidad u se llevan dos contadores, solo de los intentos en que u estaba ACTIVA:
          s_u = veces que el mundo acepto,   f_u = veces que dijo "no".
      Es una sinapsis u -> (veto del engrama) potenciada por la senal de fallo y deprimida por
      la de exito. La unidad se convierte en VETO cuando la evidencia de que "u activa => fallo"
      es fuerte: n_u = s_u + f_u >= 5 y P(p_fallo > 0.9 | s_u, f_u) > 0.95 (posterior Beta).
      No hay seleccion hacia delante ni tabla de contingencia con ausencias: cada sinapsis se
      actualiza por su cuenta con lo que ve. Eso es lo que la hace local.

    GUARDA RESULTANTE
      G = "ninguna unidad de veto esta activa". No aparece en ningun sitio la desigualdad
      hueco >= w: si emerge, es porque C-_1 y C-_2 (las unicas que pueden activarse con w <= 2)
      acumulan solo fallos, y los distractores acumulan de todo.
    """
    plastic = PLASTIC_POOLED if pooled else PLASTIC_LOCAL
    cnt = defaultdict(lambda: [0, 0])                       # u -> [s_u, f_u], solo unidades plasticas
    for c, p, w, ok in tries:
        for u, act in units(c, p, perceive_box(w)).items():
            if u not in plastic: continue
            if act: cnt[u][0 if ok else 1] += 1
    veto = [u for u, (s, f) in cnt.items() if s + f >= 5 and beta_tail(s, f) > 0.95]
    return veto, dict(cnt)

def audit(tries, veto):
    """Creencia de G ("cabe" si ninguna unidad de veto esta activa) contra lo que hizo el mundo.
       falso-abierto: G dice que cabe y el mundo dijo "no"; falso-cerrado: G dice que no y el mundo acepto."""
    fo = fc = 0
    for c, p, w, ok in tries:
        u = units(c, p, perceive_box(w)); belief = not any(u[v] for v in veto)
        fo += belief and not ok; fc += (not belief) and ok
    return len(tries), fo, fc

def exhaustive(veto, lengths=(4, 5, 6, 7, 8), widths=(1, 2, 3, 4, 5, 6)):
    """Test exhaustivo: TODAS las configuraciones binarias de estanterias de L = 4..8, todas las
    posiciones p = 0..L+1 (incluidas las de fuera) y cajas de anchura 1..6 (anchuras 3-6 nunca
    vistas en el entrenamiento). 27.264 casos. Devuelve (casos, falsos-abiertos, falsos-cerrados)."""
    n = fo = fc = 0
    for L in lengths:
        for bits in range(2 ** L):
            cells = np.array([(bits >> i) & 1 for i in range(L)], dtype=int)
            for p in range(0, L + 2):
                for w in widths:
                    ok, _ = attempt(cells, p, w)
                    u = units(cells, p, perceive_box(w)); belief = not any(u[v] for v in veto)
                    n += 1; fo += belief and not ok; fc += (not belief) and ok
    return n, fo, fc

# =============================================================================================
# 4. LA ESTRATEGIA (punto 10): otra senal de error
# =============================================================================================
def episode_fill(L, order, policy):
    """Tarea de nivel 2: llenar del todo una estanteria de L con cajas que LLEGAN EN UN ORDEN
    FIJO (`order`, cuyas anchuras suman L). No se elige la caja; solo DONDE ponerla.
    Todas las colocaciones cumplen la regla (solo se prueban casillas legales), y aun asi se
    puede acabar atascado: la regla dice que puede pasar, no que conviene.

    policy:
      "azar"            : cualquier casilla legal.
      "izquierda"       : la casilla legal mas a la izquierda (empaquetar).
      "sin_huecos_de_1" : evita dejar mas huecos de tamano 1 que cajas de 1 quedan por llegar.
    Devuelve (exito, cells): exito = la estanteria queda completamente llena.
    """
    cells = np.zeros(L, dtype=int)
    for i, w in enumerate(order):
        legal = [p for p in range(1, L + 1) if attempt(cells, p, w)[0]]
        if not legal:
            return False, cells                                  # ATASCO: la caja no cabe en ningun sitio
        rest = order[i + 1:]                                     # cajas que faltan por llegar
        if policy == "azar":
            p = int(RNG.choice(legal))
        elif policy == "izquierda":
            p = legal[0]
        elif policy == "sin_huecos_de_1":
            def leaves_bad_gap(p):
                # huecos que quedarian tras poner la caja en p
                _, nc = attempt(cells, p, w); s = "".join(map(str, nc))
                gaps = [len(x) for x in s.split("1") if x]
                # malo si hay mas huecos de tamano 1 que cajas de 1 por llegar
                return sum(1 for gp in gaps if gp == 1) > rest.count(1)
            good = [p for p in legal if not leaves_bad_gap(p)]
            p = int(RNG.choice(good if good else legal))
        cells = attempt(cells, p, w)[1]
    return bool(cells.all()), cells

def random_order(L):
    """Un orden de llegada al azar (anchuras 1 o 2) cuya suma es exactamente L."""
    while True:
        o = [int(RNG.integers(1, 3)) for _ in range(L)]
        for k in range(1, L + 1):
            if sum(o[:k]) == L:
                return o[:k]

# =============================================================================================
# 5. EJECUCION
# =============================================================================================
if __name__ == "__main__":
    # --- experiencia de entrenamiento: estanterias de 4, 5 y 6 casillas -----------------------
    trans, tries = play([4, 5, 6], episodes=400)
    print(f"Entrenamiento: estanterias de 4, 5 y 6 casillas; {len(trans)} colocaciones, {len(tries)} intentos "
          f"({sum(t[-1] for t in tries)} aceptados, {sum(not t[-1] for t in tries)} rechazados)\n")

    # --- ESCALERA: cuantos cambios distintos hay a cada nivel de canonicalizacion (puntos 4-5) -
    print("ESCALERA (cuantos cambios distintos hay)")
    for lv, name in [(0, "tal cual: en que estanteria y en que casillas"),
                     (1, "desde el ancla (desligado el DONDE)"),
                     (2, "restando la anchura (desligada la ANCHURA)")]:
        keys = Counter(canon(b, a, lv) for b, a, p, w in trans)
        print(f"  {name:<52s} {len(keys):3d}")
    print("  a mano: estanteria de 4: 4+3, de 5: 5+4, de 6: 6+5 = 27 colocaciones distintas;")
    print("  desde el ancla quedan [+1] y [+1,+1]; con la anchura como variable queda una sola regla.\n")
    lv1 = Counter(canon(b, a, 1) for b, a, p, w in trans)
    for k, n in lv1.items():
        print(f"  cambio desde el ancla {list(k)} : {n} veces")

    # --- ARQUITECTURA FIJA y ABLACION (2b) ---------------------------------------------------
    print("\nARQUITECTURA FIJA: el mismo circuito para todas las colocaciones, sin elegir nada")
    keys = Counter(canon_fixed(b, a)[0] for b, a, p, w in trans)
    print(f"  patrones normalizados distintos: {len(keys)} -> {[list(k) for k in keys]}   (un solo engrama)")
    ch = Counter(canon_fixed(b, a)[1][1] for b, a, p, w in trans)
    print(f"  canal 'w' (extension): {dict(sorted(ch.items()))}   canal 'p': donde fue la atencion")
    print("  ABLACION: que se rompe si al circuito le falta una invariancia")
    for sin, name in [("nada", "circuito completo"), ("escala", "sin reescalar (falta CUANTO)"),
                      ("atencion", "sin centrar en el cambio (falta DONDE)"), ("signo", "sin signo en el comparador")]:
        k = Counter(canon_ablation(b, a, sin) for b, a, p, w in trans)
        print(f"    {name:<42s} engramas = {len(k):3d}")
    print("  el 27 -> 2 -> 1 de arriba es esta ablacion: no una busqueda, sino lo que aporta cada invariancia.")

    # --- GUARDA: el veto se aprende de los "no", a partir del comparador (punto 8) -------------
    print("\nGUARDA: el aprendiz ve las unidades del comparador (H - W_agent), sus poblaciones N+/N- y 5 distractores.")
    veto, cnt = learn_veto(tries)
    allcnt = defaultdict(lambda: [0, 0])                      # actividad de TODAS las unidades, para la tabla
    for c, p, w, ok in tries:
        for u, act in units(c, p, perceive_box(w)).items():
            if act: allcnt[u][0 if ok else 1] += 1
    print("  actividad por unidad (exitos / fallos cuando esta activa); [plastica] = proyecta al veto")
    for u in list(comparator_units(np.zeros(6, int), 1, perceive_box(1))) + DISTRACTORES:
        s, f = allcnt.get(u, [0, 0]); pl = " [plastica]" if u in PLASTIC_POOLED else ""
        tag = "  <- VETO" if u in veto else ""
        if s + f: print(f"    {u:<24s}{pl:<12s} {s:5d} / {f:5d}{tag}")
        else:     print(f"    {u:<24s}{pl:<12s} nunca activa")
    print("  G = 'ninguna unidad de veto activa'. Vetos aprendidos:", ", ".join(veto))
    print("  lectura: 'hueco >= w' no se ha escrito en ningun sitio. N- (hay mismatch negativo en alguna posicion)")
    print("  coincide solo con fallos: 0 / 3761. El aprendiz lo convierte en veto. Los distractores, no.")
    n, fo, fc = audit(tries, veto)
    print(f"  auditoria (entrenamiento): {n} intentos, falsos-abiertos {fo}, falsos-cerrados {fc}")
    print("  ejemplo: hueco 1, caja 2:  H = 100000,  W = 110000,  H - W = 0 -1 0 0 0 0  -> C-_2 y N- activas -> veto")

    # --- TEST EXHAUSTIVO: anchuras nunca vistas (3-6) con los vetos congelados -----------------
    print("\nTEST EXHAUSTIVO: L = 4..8, todas las configuraciones, p = 0..L+1, anchuras 1..6 (3-6 nunca vistas)")
    n, fo, fc = exhaustive(veto)
    print(f"  vetos aprendidos sobre N- (poblacion convergente):   {n} casos, falsos-abiertos {fo}, falsos-cerrados {fc}")
    veto_pp, _ = learn_veto(tries, pooled=False)
    n, fo, fc = exhaustive(veto_pp)
    print(f"  ablacion, vetos por posicion ({', '.join(veto_pp)}): {n} casos, falsos-abiertos {fo}, falsos-cerrados {fc}")
    print("  ejemplo: caja 3 con hueco 2: H = 110000, W = 111000, H - W = 0 0 -1 0 0 0: solo C-_3, que nunca se")
    print("  entreno, no veta; N- si. La transferencia es a CANTIDADES nuevas, no solo a estanterias nuevas.")

    # --- EVIDENCIA Theta: cuanto me fio de la regla tras n intentos (punto 6) -----------------
    print("\nEVIDENCIA: cuanto me fio de 'sin veto, cabe' tras n intentos (posterior Beta(1+exitos, 1+fallos))")
    ok_rows = [ok for c, p, w, ok in tries if not any(units(c, p, perceive_box(w))[v] for v in veto)]
    for n in (3, 10, 30, 300):
        s = sum(ok_rows[:n]); f = n - s
        a, b = 1 + s, 1 + f
        mean = a / (a + b); sd = (a * b / ((a + b) ** 2 * (a + b + 1))) ** 0.5
        print(f"  n={n:4d}: exitos {s:3d}, fallos {f:2d} -> p = {mean:.2f} +- {sd:.2f}")

    # --- TRANSFERENCIA: estanterias nunca vistas, regla congelada (punto 9) -------------------
    trans7, tries7 = play([7, 8], episodes=200)
    n, fo, fc = audit(tries7, veto)
    print(f"\nTRANSFERENCIA a estanterias de 7 y 8 casillas, nunca vistas, con la regla congelada")
    print(f"  {n} intentos: falsos-abiertos {fo}, falsos-cerrados {fc}")
    # control: quien memorizo (L, p, w) como hechos sueltos no reconoce nada en una L nueva
    seen = {(len(c), p, w) for c, p, w, ok in tries if ok}
    ok7 = [(len(c), p, w) for c, p, w, ok in tries7 if ok]
    print(f"  control (tabla memorizada por estanteria, casilla y anchura): reconoce {sum(k in seen for k in ok7)}/{len(ok7)} "
          f"colocaciones legales de las estanterias nuevas")
    new = Counter(canon(b, a, 1) for b, a, p, w in trans7)
    print(f"  cambios desde el ancla en las nuevas: {[list(k) for k in new]}  (los mismos dos)")

    # --- ESTRATEGIA: nivel 2, se aprende de los atascos (punto 10) ---------------------------
    print("\nESTRATEGIA: llenar una estanteria de 6 con cajas que llegan en un orden fijo (solo eliges DONDE)")
    print("  Las reglas nunca dicen 'no'; aun asi se puede acabar atascado. Exito = estanteria llena.")
    orders = [random_order(6) for _ in range(500)]
    for pol in ("azar", "izquierda", "sin_huecos_de_1"):
        ok = sum(episode_fill(6, o, pol)[0] for o in orders)
        print(f"  {pol:<16s} {ok}/500 llenas")
    print("  ejemplo: llegan [1, 1, 2, 2]; poner los 1 en las casillas 2 y 4 deja huecos 1,1,2: la ultima caja de 2 no cabe.")
    # Aprender la estrategia de los atascos: el mismo tipo de tabla de conteo que la guarda,
    # pero (i) el resultado es "alcance el objetivo" y no "el mundo acepto", y (ii) el rasgo es
    # sobre un estado mas abstracto (la lista de huecos), no sobre el estado casilla a casilla.
    s_a = f_a = s_p = f_p = 0
    for o in orders:
        L = 6; cells = np.zeros(L, dtype=int); flagged = False; stuck = False
        for i, w in enumerate(o):
            legal = [p for p in range(1, L + 1) if attempt(cells, p, w)[0]]
            if not legal:
                stuck = True; break
            p = int(RNG.choice(legal)); cells = attempt(cells, p, w)[1]
            s = "".join(map(str, cells)); gaps = [len(x) for x in s.split("1") if x]
            if sum(1 for gp in gaps if gp == 1) > o[i + 1:].count(1):
                flagged = True                                   # el rasgo: "deje un hueco de 1 sin caja de 1 por llegar"
        if flagged: f_p += stuck; s_p += not stuck
        else:       f_a += stuck; s_a += not stuck
    print(f"  aprendida de los atascos: 'dejo un hueco de 1 sin caja de 1 por llegar'  "
          f"ausente: llenas {s_a}, atascos {f_a} | presente: llenas {s_p}, atascos {f_p}")
    print("\nPARIDAD (razonar con las reglas): una estanteria de 7 solo se llena con un numero IMPAR de cajas de 1.")
