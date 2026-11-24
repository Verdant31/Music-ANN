"""Pares de controle: original x original com a cadeia lo-fi sintética.

Grava dados/sintetico/<nome>.npz no mesmo formato de preparar_dados.py, com
alinhamento perfeito (mapa = identidade).

    python preparar_sintetico.py
    python treinar.py --pares dados/sintetico --saida modelos/sintetico
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from musicann import audio
from musicann.efeito_lofi import aplicar


def processar(arquivo, pasta_saida, semente):
    nome = os.path.splitext(os.path.basename(arquivo))[0]
    y = audio.carregar(arquivo)
    y_l, _ = aplicar(y, np.random.default_rng(semente))
    m_o, m_l = audio.mel_db(y), audio.mel_db(y_l)
    T = m_l.shape[1]
    np.savez_compressed(
        os.path.join(pasta_saida, nome + ".npz"),
        original=m_o, lofi=m_l, mapa=np.arange(T), custo=np.zeros(T, np.float32),
        inclinacao=np.ones(T, np.float32), tom=np.float32(0),
    )
    return nome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicas", default="Musics/Normal")
    ap.add_argument("--saida", default="dados/sintetico")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)
    arquivos = sorted(os.path.join(args.musicas, f) for f in os.listdir(args.musicas) if f.endswith(".mp3"))
    with ProcessPoolExecutor(6) as ex:
        for nome in ex.map(processar, arquivos, [args.saida] * len(arquivos), range(len(arquivos))):
            print(nome)


if __name__ == "__main__":
    main()
