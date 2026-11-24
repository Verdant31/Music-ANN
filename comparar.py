"""Figura e áudios de comparação num trecho de uma música de teste.

Para uma janela de ~12 s mostra lado a lado: a original (alinhada), a lo-fi
de verdade, o "EQ lo-fi" fixo e a previsão da U-Net. Também grava os quatro
em .wav, todos passados pelo mesmo Mel -> áudio, para ouvir em pé de
igualdade.

    python comparar.py Hawayein --inicio 60
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from musicann import audio
from musicann.dados import carregar_pares
from gerar_lofi import carregar_modelo, transformar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("musica", nargs="?")
    ap.add_argument("--inicio", type=float, default=60.0, help="segundos, na linha do tempo da lo-fi")
    ap.add_argument("--duracao", type=float, default=12.0)
    ap.add_argument("--modelo", default="modelos/unet.pt")
    ap.add_argument("--pares", default="dados/pares")
    ap.add_argument("--saida", default="exemplos")
    args = ap.parse_args()

    if args.musica is None:
        with open(os.path.join(os.path.dirname(args.modelo), "resultado.json"), encoding="utf-8") as f:
            args.musica = json.load(f)["teste"][0]
    par = carregar_pares(args.pares)[args.musica]
    q = lambda s: int(s * audio.SR / audio.HOP)
    a, b = q(args.inicio), q(args.inicio + args.duracao)
    mapa = np.clip(par["mapa"], 0, par["original"].shape[1] - 1)
    original = par["original"][:, mapa[a:b]]
    lofi = par["lofi"][:, a:b]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.modelo, map_location=dev)
    eq = audio.desnormalizar(audio.normalizar(original) + ck["eq"].cpu().numpy()[0])
    unet = transformar(carregar_modelo(args.modelo, dev), original, dev)

    versoes = {"original (alinhada)": original, "lo-fi real": lofi, "EQ lo-fi fixo": eq, "U-Net": unet}
    os.makedirs(args.saida, exist_ok=True)
    fig, eixos = plt.subplots(len(versoes), 1, figsize=(10, 9), sharex=True)
    for ax, (titulo, db) in zip(eixos, versoes.items()):
        ax.imshow(np.clip(db, -80, 20), origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=20)
        ax.set_ylabel("banda Mel")
        ax.set_title(titulo, loc="left", fontsize=10)
    eixos[-1].set_xlabel("quadro (~23 ms)")
    fig.suptitle(f"{args.musica.replace('_', ' ')}: {args.inicio:.0f}s a {args.inicio + args.duracao:.0f}s da lo-fi")
    fig.tight_layout()
    sufixo = "" if args.pares == "dados/pares" else "_" + os.path.basename(args.pares.rstrip("/"))
    args.musica_arq = args.musica + sufixo
    figura = os.path.join(args.saida, f"{args.musica_arq}.png")
    fig.savefig(figura, dpi=110)
    print(f"gravado: {figura}")

    for titulo, db in versoes.items():
        y = audio.mel_para_audio(db, iteracoes=64)
        nome = titulo.split(" (")[0].replace(" ", "_").replace("-", "")
        caminho = os.path.join(args.saida, f"{args.musica_arq}_{nome}.wav")
        sf.write(caminho, 0.95 * y / (np.abs(y).max() + 1e-8), audio.SR)
        print(f"gravado: {caminho}")


if __name__ == "__main__":
    main()
