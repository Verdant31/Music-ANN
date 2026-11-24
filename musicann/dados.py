"""Etapa 2: transforma os pares alinhados em janelas de treino."""
import glob
import os

import numpy as np

from .audio import normalizar

JANELA = 128  # quadros (~3 s)


def carregar_pares(pasta="dados/pares"):
    pares = {}
    for f in sorted(glob.glob(os.path.join(pasta, "*.npz"))):
        pares[os.path.splitext(os.path.basename(f))[0]] = dict(np.load(f))
    return pares


CUSTO_FINO_MAX = None  # definido por treinar.py (--custo-fino-max)


def quadros_validos(par, custo_max, incl_min=0.6, incl_max=1.6):
    c, inc = par["custo"], par["inclinacao"]
    ok = np.isfinite(c) & (c <= custo_max) & (inc >= incl_min) & (inc <= incl_max)
    if CUSTO_FINO_MAX is not None and "custo_fino" in par:
        ok &= par["custo_fino"] <= CUSTO_FINO_MAX
    return ok


def janelas(par, custo_max, passo=32, janela=JANELA):
    """Devolve (X, Y): original alinhada e lo-fi, já normalizadas, (N, 128, janela)."""
    ok = quadros_validos(par, custo_max)
    mapa = np.clip(par["mapa"], 0, par["original"].shape[1] - 1)
    T = par["lofi"].shape[1]
    X, Y = [], []
    for s in range(0, T - janela + 1, passo):
        if ok[s : s + janela].all():
            X.append(normalizar(par["original"][:, mapa[s : s + janela]]))
            Y.append(normalizar(par["lofi"][:, s : s + janela]))
    if not X:
        return np.zeros((0, 128, janela), np.float32), np.zeros((0, 128, janela), np.float32)
    return np.stack(X), np.stack(Y)


def montar(pares, nomes, custo_max, passo=32):
    X, Y = zip(*(janelas(pares[n], custo_max, passo) for n in nomes))
    return np.concatenate(X), np.concatenate(Y)
