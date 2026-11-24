"""Rede que aprende, com os pares reais, os ajustes da montagem lo-fi.

Entrada: características da música original (espectro médio e variação por
faixa, andamento, brilho). Saída: velocidade, tom, corte e graves, os
mesmos parâmetros de musicann/producao.Ajustes. Com ~60 músicas a rede é
pequena de propósito (uma camada escondida, dropout e weight decay).
"""
import numpy as np
import torch
import torch.nn as nn

ALVOS = ["velocidade", "tom", "corte", "graves_db"]
LIMITES = {"velocidade": (0.75, 1.0), "tom": (-3.0, 0.5), "corte": (900.0, 11000.0), "graves_db": (-10.0, 10.0)}


def preparar_alvos(Y, nomes_alvos):
    """Seleciona os alvos usados, limita os valores e põe o corte em log."""
    idx = [list(nomes_alvos).index(a) for a in ALVOS]
    Y = Y[:, idx].copy()
    for j, a in enumerate(ALVOS):
        Y[:, j] = np.clip(Y[:, j], *LIMITES[a])
    Y[:, ALVOS.index("corte")] = np.log(Y[:, ALVOS.index("corte")])
    return Y


class RedeDeAjustes(nn.Module):
    def __init__(self, n_entrada, n_saida=len(ALVOS), oculta=32, dropout=0.3):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(n_entrada, oculta), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(oculta, n_saida))

    def forward(self, x):
        return self.rede(x)


def treinar(X, Y, epocas=400, lr=3e-3, decaimento=1e-2, semente=0):
    """Treina com entradas e alvos padronizados; devolve um preditor."""
    torch.manual_seed(semente)
    mx, sx = X.mean(0), X.std(0) + 1e-6
    my, sy = Y.mean(0), Y.std(0) + 1e-6
    xt = torch.tensor((X - mx) / sx, dtype=torch.float32)
    yt = torch.tensor((Y - my) / sy, dtype=torch.float32)
    rede = RedeDeAjustes(X.shape[1], Y.shape[1])
    opt = torch.optim.AdamW(rede.parameters(), lr=lr, weight_decay=decaimento)
    for _ in range(epocas):
        rede.train()
        perda = nn.functional.smooth_l1_loss(rede(xt), yt)
        opt.zero_grad()
        perda.backward()
        opt.step()
    rede.eval()
    return dict(rede=rede, mx=mx, sx=sx, my=my, sy=sy)


def prever(modelo, x):
    with torch.no_grad():
        z = modelo["rede"](torch.tensor((np.atleast_2d(x) - modelo["mx"]) / modelo["sx"], dtype=torch.float32))
    y = z.numpy() * modelo["sy"] + modelo["my"]
    return y


def para_ajustes(y):
    """Converte uma linha prevista em kwargs de producao.Ajustes."""
    d = dict(zip(ALVOS, np.asarray(y).ravel().tolist()))
    d["corte"] = float(np.exp(d["corte"]))
    for a in ALVOS:
        d[a] = float(np.clip(d[a], *LIMITES[a]))
    return dict(velocidade=d["velocidade"], manter_tom=d["tom"] > -0.5, corte_base=d["corte"],
                graves_db=min(0.0, d["graves_db"]), tom=d["tom"])
