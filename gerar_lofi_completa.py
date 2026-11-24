"""Gera a versão lo-fi completa de uma música.

  1. Separa voz e instrumental (Demucs).
  2. Voz: a U-Net treinada em pares reais (voz original -> voz lo-fi) diz
     como cada faixa de frequência muda; aplico essa mudança no próprio áudio
     da voz, mantendo a fase (som limpo, sem Griffin-Lim).
  3. Instrumental, uma de duas opções:
       --batida original  o instrumental da música, tratado por uma U-Net
                          treinada nos instrumentais (ou só o EQ dela)
       --batida musicgen  uma base lo-fi nova gerada pelo MusicGen-melody,
                          seguindo os acordes do instrumental original
  4. Desacelera tudo (--velocidade) e mistura.

    python gerar_lofi_completa.py musica.mp3 --batida musicgen --velocidade 0.88
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from musicann import audio
from gerar_lofi import carregar_modelo, transformar


def tratar(y, caminho_modelo, dev):
    db = audio.mel_db(y)
    pred = transformar(carregar_modelo(caminho_modelo, dev), db, dev)
    return audio.aplicar_diferenca(y, db, pred)


def normalizar_volume(y, alvo_db=-20.0):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    return y * 10 ** (alvo_db / 20) / rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--saida")
    ap.add_argument("--modelo-voz", default="modelos/voz/unet.pt")
    ap.add_argument("--modelo-inst", default="modelos/inst/unet.pt")
    ap.add_argument("--batida", choices=["original", "musicgen"], default="original")
    ap.add_argument("--velocidade", type=float, default=0.88)
    ap.add_argument("--volume-voz", type=float, default=0.0, help="dB a mais na voz em relação à base")
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    saida = args.saida or os.path.splitext(os.path.basename(args.entrada))[0] + f"_lofi_{args.batida}.wav"

    from demucs.pretrained import get_model
    from separar import separar
    demucs = get_model("htdemucs").to(dev).eval()
    voz, inst = separar(demucs, args.entrada, dev)
    del demucs
    torch.cuda.empty_cache()

    voz = tratar(voz, args.modelo_voz, dev)
    if args.batida == "original":
        if os.path.exists(args.modelo_inst):
            inst = tratar(inst, args.modelo_inst, dev)
    if args.velocidade != 1.0:
        voz = librosa.effects.time_stretch(voz, rate=args.velocidade)
        inst = librosa.effects.time_stretch(inst, rate=args.velocidade)
    if args.batida == "musicgen":
        from musicann.batida import GeradorDeBatida
        bpm = float(np.atleast_1d(librosa.beat.beat_track(y=inst, sr=audio.SR)[0])[0])
        if bpm > 110:
            bpm /= 2  # lo-fi costuma ficar entre 70 e 95 bpm
        print(f"andamento estimado: {bpm:.0f} bpm")
        inst = GeradorDeBatida(dev).gerar(inst, audio.SR, bpm, semente=args.semente)

    n = min(len(voz), len(inst))
    mix = normalizar_volume(voz[:n], -20 + args.volume_voz) + normalizar_volume(inst[:n], -21)
    mix = 0.95 * mix / (np.abs(mix).max() + 1e-8)
    sf.write(saida, mix, audio.SR)
    print(f"gravado: {saida}")


if __name__ == "__main__":
    main()
