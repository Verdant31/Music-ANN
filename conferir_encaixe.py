"""Confere, trecho a trecho, se a bateria da lo-fi gerada bate com a música.

Separa a bateria da lo-fi gerada e a da original (Demucs), leva a original
para a mesma velocidade e mede, em janelas de 20 s, a correlação dos ataques
das duas (com tolerância de ~1/8 de batida). Janelas com correlação baixa são
trechos em que a bateria nova pode ter saído do tempo.

    python conferir_encaixe.py exemplos/completas/Kendrick_Money_Trees.wav Musics/Normal/Kendrick_Money_Trees.mp3
"""
import argparse

import librosa
import numpy as np

from musicann import producao as P
from musicann.producao import HOP_ENC, _ataques, _correlacao


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gerada")
    ap.add_argument("original")
    ap.add_argument("--velocidade", type=float, default=P.Ajustes().velocidade)
    ap.add_argument("--janela", type=float, default=20.0)
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from separar import separar4
    modelo = get_model("htdemucs").to("cuda").eval()
    g, _ = librosa.load(args.gerada, sr=P.SR, mono=True)
    o, _ = librosa.load(args.original, sr=P.SR, mono=True)
    bat_g = separar4(modelo, g, P.SR, "cuda")["drums"]
    bat_o = P.desacelerar(separar4(modelo, o, P.SR, "cuda")["drums"], args.velocidade)
    a_g, a_o = _ataques(bat_g), _ataques(bat_o)
    n = min(len(a_g), len(a_o))
    q = int(args.janela * P.SR / HOP_ENC)
    tol = 3  # quadros (~35 ms)
    notas = []
    for ini in range(0, n - q // 2, q):
        x, y = a_o[ini : ini + q], a_g[ini : ini + q]
        if np.std(x) < 1e-3 or np.std(y) < 1e-3:
            notas.append(np.nan)
            continue
        notas.append(max(_correlacao(x, y, d) for d in range(-tol, tol + 1)))
    notas = np.array(notas)
    linha = " ".join("  -- " if np.isnan(v) else f"{v:4.2f}" for v in notas)
    fracos = [f"{int(i * args.janela // 60)}:{int(i * args.janela % 60):02d}" for i, v in enumerate(notas) if v < 0.2]
    print(f"encaixe a cada {args.janela:.0f}s: {linha}")
    print(f"média {np.nanmean(notas):.2f} | trechos fracos (< 0,2): {', '.join(fracos) or 'nenhum'}")


if __name__ == "__main__":
    main()
