"""Treina a rede de ajustes e mede se ela ganha de chutar a média.

Validação "deixa uma de fora": para cada música, treina com todas as outras
e prevê os ajustes dela. Compara o erro médio com duas referências:
  - média: prever sempre a média das outras músicas
  - linear: regressão ridge sobre as mesmas características
No fim treina com todas e grava modelos/ajustes.pt.

    python medir_ajustes.py && python treinar_ajustes.py
"""
import numpy as np
import torch

from musicann.ajustes import ALVOS, prever, preparar_alvos, treinar


def ridge(Xtr, Ytr, Xte, alfa=10.0):
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mx) / sx
    my = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + alfa * np.eye(A.shape[1]), A.T @ (Ytr - my))
    return ((Xte - mx) / sx) @ W + my


def main():
    d = np.load("dados/ajustes.npz")
    X, Y, nomes = d["X"], preparar_alvos(d["Y"], d["alvos"]), d["nomes"]
    n = len(X)
    erros = {"média": [], "linear": [], "rede": []}
    for i in range(n):
        tr = np.arange(n) != i
        erros["média"].append(np.abs(Y[tr].mean(0) - Y[i]))
        erros["linear"].append(np.abs(ridge(X[tr], Y[tr], X[i : i + 1])[0] - Y[i]))
        prev = np.mean([prever(treinar(X[tr], Y[tr], semente=s), X[i])[0] for s in range(3)], axis=0)
        erros["rede"].append(np.abs(prev - Y[i]))
    print(f"{n} músicas, validação deixa-uma-de-fora. Erro médio por ajuste (corte em log):")
    print(f"{'':8} " + " ".join(f"{a:>11}" for a in ALVOS))
    for k, v in erros.items():
        print(f"{k:8} " + " ".join(f"{x:11.3f}" for x in np.mean(v, 0)))
    ganho = 1 - np.mean(erros["rede"], 0) / np.mean(erros["média"], 0)
    print("rede vs média: " + " ".join(f"{a} {g:+.0%}" for a, g in zip(ALVOS, ganho)))

    modelo = treinar(X, Y)
    torch.save(dict(estado=modelo["rede"].state_dict(), mx=modelo["mx"], sx=modelo["sx"], my=modelo["my"],
                    sy=modelo["sy"], n_entrada=X.shape[1], ganho=ganho.tolist()), "modelos/ajustes.pt")
    print("gravado: modelos/ajustes.pt")


if __name__ == "__main__":
    main()
