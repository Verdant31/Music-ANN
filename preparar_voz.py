"""Alinha os pares usando só a voz separada pelo Demucs.

Na lo-fi a batida é nova, mas a voz é a da original (desacelerada, filtrada,
com reverb). Alinhando pela voz, a batida nova deixa de atrapalhar.

Grava três conjuntos com o mesmo mapa de alinhamento:
  dados/voz/<nome>.npz   voz original x voz lo-fi
  dados/inst/<nome>.npz  instrumental original x instrumental lo-fi
  dados/mix/<nome>.npz   mixagem original x lo-fi completa

    python separar.py && python preparar_voz.py
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np

from musicann import audio
from musicann.alinhamento import alinhar, chroma, deslocamento_de_tom, feicao_harmonica, refinar
from preparar_dados import correlacao


def carregar(pasta, lado, nome, parte):
    return audio.carregar(os.path.join(pasta, "Separado", lado, f"{nome}_{parte}.wav"))


def processar(nome, musicas, saida):
    v_o, v_l = carregar(musicas, "Normal", nome, "voz"), carregar(musicas, "Lofi", nome, "voz")
    i_o, i_l = carregar(musicas, "Normal", nome, "inst"), carregar(musicas, "Lofi", nome, "inst")

    c_o, c_l = chroma(v_o), chroma(v_l)
    rms_l = librosa.feature.rms(y=v_l, frame_length=audio.N_FFT, hop_length=audio.HOP)[0]
    k = deslocamento_de_tom(c_o, c_l, rms_l)
    tom = k + librosa.estimate_tuning(y=v_l, sr=audio.SR) - librosa.estimate_tuning(y=v_o, sr=audio.SR)
    if abs(tom) > 0.3:
        v_o, i_o = (librosa.effects.pitch_shift(s, sr=audio.SR, n_steps=tom) for s in (v_o, i_o))
        c_o = chroma(v_o)

    mapa, custo, incl = alinhar(c_o, c_l, rms_l)
    mv_o, mv_l = audio.mel_db(v_o), audio.mel_db(v_l)
    mapa_bruto = mapa
    mapa, custo_fino = refinar(feicao_harmonica(mv_o), feicao_harmonica(mv_l), mapa, custo)

    n_o, n_l = min(len(v_o), len(i_o)), min(len(v_l), len(i_l))
    comum = dict(mapa=mapa, mapa_bruto=mapa_bruto, custo=custo, custo_fino=custo_fino,
                 inclinacao=incl, tom=np.float32(tom))
    for parte, (a, b) in {
        "voz": (mv_o, mv_l),
        "inst": (audio.mel_db(i_o), audio.mel_db(i_l)),
        "mix": (audio.mel_db(v_o[:n_o] + i_o[:n_o]), audio.mel_db(v_l[:n_l] + i_l[:n_l])),
    }.items():
        np.savez_compressed(os.path.join(saida, parte, nome + ".npz"), original=a, lofi=b, **comum)

    ok = np.isfinite(custo)
    return dict(nome=nome, tom=tom,
                incl=float(np.median(incl[ok])) if ok.any() else float("nan"),
                acaso=correlacao(mv_o, mv_l, mapa + 200, ok), corr=correlacao(mv_o, mv_l, mapa, ok),
                validos=float(ok.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicas", default="Musics")
    ap.add_argument("--saida", default="dados")
    args = ap.parse_args()
    for parte in ["voz", "inst", "mix"]:
        os.makedirs(os.path.join(args.saida, parte), exist_ok=True)
    pasta = os.path.join(args.musicas, "Separado", "Lofi")
    nomes = sorted(f[: -len("_voz.wav")] for f in os.listdir(pasta) if f.endswith("_voz.wav")
                   if os.path.exists(os.path.join(args.musicas, "Separado", "Normal", f)))
    print(f"{len(nomes)} pares")
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(processar, nomes, [args.musicas] * len(nomes), [args.saida] * len(nomes)))
    print(f"\n{'música':32} {'tom':>6} {'veloc':>6} {'voz%':>5} {'acaso':>6} {'corr':>6}")
    for r in sorted(res, key=lambda r: -r["corr"]):
        print(f"{r['nome']:32} {r['tom']:+6.2f} {r['incl']:6.2f} {r['validos']:5.0%} {r['acaso']:6.3f} {r['corr']:6.3f}")


if __name__ == "__main__":
    main()
