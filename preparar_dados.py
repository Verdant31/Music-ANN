"""Etapa 1: espectrogramas e alinhamento de cada par.

Lê Musics/Normal/<nome>.mp3 e Musics/Lofi/<nome>.mp3 e grava
dados/pares/<nome>.npz com os log-Mel das duas versões e o mapa de
alinhamento. No fim imprime um resumo por música.

    python preparar_dados.py
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

from musicann import audio
from musicann.alinhamento import alinhar, chroma, deslocamento_de_tom, feicao_harmonica, refinar


def correlacao(m_o, m_l, mapa, ok):
    """Conferência independente do chroma: correlação média, quadro a quadro,
    entre os espectrogramas (bandas 10-90) da original alinhada e da lo-fi."""
    if not ok.any():
        return float("nan")
    a = m_o[10:90, np.clip(mapa[ok], 0, m_o.shape[1] - 1)]
    b = m_l[10:90, ok]
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)
    b = (b - b.mean(0)) / (b.std(0) + 1e-6)
    return float((a * b).mean())


def processar(nome, pasta_musicas, pasta_saida):
    y_o = audio.carregar(os.path.join(pasta_musicas, "Normal", nome + ".mp3"))
    y_l = audio.carregar(os.path.join(pasta_musicas, "Lofi", nome + ".mp3"))

    c_o, c_l = chroma(y_o), chroma(y_l)
    rms_l = librosa.feature.rms(y=y_l, frame_length=audio.N_FFT, hop_length=audio.HOP)[0]
    k = deslocamento_de_tom(c_o, c_l, rms_l)
    afin = librosa.estimate_tuning(y=y_l, sr=audio.SR) - librosa.estimate_tuning(y=y_o, sr=audio.SR)
    tom = k + afin
    if abs(tom) > 0.3:
        # Leva a original para o tom da lo-fi, assim a rede não precisa
        # aprender a transpor, só a mudar o timbre.
        y_o = librosa.effects.pitch_shift(y_o, sr=audio.SR, n_steps=tom)
        c_o = chroma(y_o)

    mapa, custo, incl = alinhar(c_o, c_l, rms_l)

    m_o, m_l = audio.mel_db(y_o), audio.mel_db(y_l)
    mapa_bruto = mapa
    mapa, custo_fino = refinar(feicao_harmonica(m_o), feicao_harmonica(m_l), mapa, custo)
    np.savez_compressed(
        os.path.join(pasta_saida, nome + ".npz"),
        original=m_o, lofi=m_l, mapa_bruto=mapa_bruto,
        mapa=mapa, custo=custo, custo_fino=custo_fino, inclinacao=incl, tom=np.float32(tom),
    )
    ok = np.isfinite(custo)
    return dict(
        corr=correlacao(m_o, m_l, mapa, ok), corr_acaso=correlacao(m_o, m_l, mapa + 200, ok),
        corr_bruto=correlacao(m_o, m_l, mapa_bruto, ok),
        nome=nome, dur_o=len(y_o) / audio.SR, dur_l=len(y_l) / audio.SR, tom=tom,
        custo=float(np.median(custo[ok])) if ok.any() else float("nan"),
        incl=float(np.median(incl[ok])) if ok.any() else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicas", default="Musics")
    ap.add_argument("--saida", default="dados/pares")
    ap.add_argument("--processos", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.saida, exist_ok=True)
    nomes = sorted(
        os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.musicas, "Lofi"))
        if f.endswith(".mp3") and os.path.exists(os.path.join(args.musicas, "Normal", f))
    )
    print(f"{len(nomes)} pares")
    with ProcessPoolExecutor(args.processos) as ex:
        res = list(ex.map(processar, nomes, [args.musicas] * len(nomes), [args.saida] * len(nomes)))

    print(f"\n{'música':32} {'orig':>6} {'lofi':>6} {'tom':>6} {'custo':>6} {'veloc':>6} "
          f"{'acaso':>6} {'chroma':>6} {'fino':>6}")
    for r in sorted(res, key=lambda r: r["custo"]):
        print(f"{r['nome']:32} {r['dur_o']:6.0f} {r['dur_l']:6.0f} {r['tom']:+6.2f} {r['custo']:6.3f} "
              f"{r['incl']:6.2f} {r['corr_acaso']:6.3f} {r['corr_bruto']:6.3f} {r['corr']:6.3f}")


if __name__ == "__main__":
    main()
