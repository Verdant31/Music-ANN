"""Confere se a original alinhada anda junto com a lo-fi.

1. Áudio: grava um .wav estéreo com a original alinhada no canal esquerdo e
   a lo-fi no direito. Se o alinhamento estiver bom, as duas vozes andam
   juntas; se não, embolam.
2. Medida: para cada bloco de ~3 s, procura o deslocamento (em quadros de
   ~23 ms) que melhor casa os dois espectrogramas. Com alinhamento perfeito
   o melhor deslocamento é 0.

    python conferir_alinhamento.py Muskurane --inicio 60
    python conferir_alinhamento.py --todas
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf

from musicann import audio
from musicann.dados import carregar_pares, quadros_validos

MAX_DESLOC = 24  # quadros (~0,55 s)


def melhor_deslocamento(par, custo_max=0.2, bloco=128):
    """Lista com o melhor deslocamento de cada bloco válido."""
    ok = quadros_validos(par, custo_max)
    mapa = np.clip(par["mapa"], 0, par["original"].shape[1] - 1)
    o, l = par["original"][10:90], par["lofi"][10:90]
    res = []
    for s in range(MAX_DESLOC, l.shape[1] - bloco - MAX_DESLOC, bloco):
        if not ok[s : s + bloco].all():
            continue
        b = l[:, s : s + bloco]
        b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)
        corr = []
        for d in range(-MAX_DESLOC, MAX_DESLOC + 1):
            idx = np.clip(mapa[s : s + bloco] + d, 0, o.shape[1] - 1)
            a = o[:, idx]
            a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
            corr.append((a * b).mean())
        res.append(int(np.argmax(corr)) - MAX_DESLOC)
    return res


def audio_alinhado(nome, par, inicio, duracao, musicas="Musics"):
    """Reconstrói a original na linha do tempo da lo-fi (soma de janelas)."""
    y_o = audio.carregar(os.path.join(musicas, "Normal", nome + ".mp3"))
    if abs(float(par["tom"])) > 0.3:
        y_o = librosa.effects.pitch_shift(y_o, sr=audio.SR, n_steps=float(par["tom"]))
    y_l = audio.carregar(os.path.join(musicas, "Lofi", nome + ".mp3"))
    a = int(inicio * audio.SR / audio.HOP)
    n = int(duracao * audio.SR / audio.HOP)
    N, H = audio.N_FFT, audio.HOP
    jan = np.hanning(N)
    saida = np.zeros((n + 4) * H + N)
    soma = np.zeros_like(saida)
    y_o = np.pad(y_o, (N, N))
    for k in range(n):
        c = par["mapa"][a + k] * H + N  # centro do quadro na original (com o pad)
        pedaco = y_o[c - N // 2 : c + N // 2]
        if len(pedaco) == N:
            saida[k * H : k * H + N] += pedaco * jan
            soma[k * H : k * H + N] += jan
    saida = saida / np.maximum(soma, 1e-3)
    lofi = y_l[a * H : a * H + len(saida)]
    m = min(len(saida), len(lofi))
    estereo = np.stack([saida[:m], lofi[:m]], axis=1)
    return 0.95 * estereo / (np.abs(estereo).max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("musica", nargs="?")
    ap.add_argument("--inicio", type=float, default=60.0)
    ap.add_argument("--duracao", type=float, default=20.0)
    ap.add_argument("--todas", action="store_true", help="só a medida, para todas as músicas")
    ap.add_argument("--pares", default="dados/pares")
    args = ap.parse_args()

    pares = carregar_pares(args.pares)
    nomes = sorted(pares) if args.todas else [args.musica]
    todos = []
    for nome in nomes:
        d = np.array(melhor_deslocamento(pares[nome]))
        todos.extend(d.tolist())
        if len(d):
            print(f"{nome:32} blocos {len(d):3d}  desloc. 0: {np.mean(d == 0):4.0%}  "
                  f"|d|<=2: {np.mean(np.abs(d) <= 2):4.0%}  mediana |d|: {np.median(np.abs(d)):4.1f}")
    d = np.array(todos)
    print(f"\nTOTAL blocos {len(d)}  desloc. 0: {np.mean(d == 0):.0%}  |d|<=2: {np.mean(np.abs(d) <= 2):.0%}  "
          f"|d|>=10: {np.mean(np.abs(d) >= 10):.0%}")

    if not args.todas:
        os.makedirs("exemplos", exist_ok=True)
        caminho = os.path.join("exemplos", f"{args.musica}_alinhamento.wav")
        sf.write(caminho, audio_alinhado(args.musica, pares[args.musica], args.inicio, args.duracao), audio.SR)
        print(f"gravado: {caminho} (esquerda: original alinhada, direita: lo-fi)")


if __name__ == "__main__":
    main()
