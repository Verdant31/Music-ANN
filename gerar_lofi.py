"""Gera a versão lo-fi de qualquer música com a U-Net treinada.

    python gerar_lofi.py minha_musica.mp3
    python gerar_lofi.py minha_musica.mp3 --saida saida.wav --velocidade 0.9

A rede só muda o timbre. Deixar mais lento é um efeito à parte
(--velocidade), aplicado depois.
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch

from musicann import audio
from musicann.modelo import UNet, completar


def carregar_modelo(caminho, dev):
    ck = torch.load(caminho, map_location=dev)
    modelo = UNet(base=ck["base"]).to(dev).eval()
    modelo.load_state_dict(ck["estado"])
    modelo.volume_relativo = ck.get("volume_relativo", False)
    return modelo


@torch.no_grad()
def transformar(modelo, db, dev):
    x = torch.from_numpy(audio.normalizar(db))[None].to(dev)
    media = x.mean() if getattr(modelo, "volume_relativo", False) else 0.0
    x, falta = completar(x - media, modelo.multiplo)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = modelo(x).float() + media
    if falta:
        p = p[..., :-falta]
    return audio.desnormalizar(p[0].cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--saida")
    ap.add_argument("--modelo", default="modelos/unet.pt")
    ap.add_argument("--velocidade", type=float, default=1.0, help="ex.: 0.9 = 10%% mais lenta")
    ap.add_argument("--iteracoes", type=int, default=64)
    ap.add_argument("--referencia", action="store_true",
                    help="também grava a original passada pelo mesmo Mel -> áudio, para comparar")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    saida = args.saida or os.path.splitext(os.path.basename(args.entrada))[0] + "_lofi.wav"
    y = audio.carregar(args.entrada)
    fase = audio.stft(y)
    db = audio.mel_db(y)

    pred = transformar(carregar_modelo(args.modelo, dev), db, dev)
    y_lofi = audio.mel_para_audio(pred, fase_inicial=fase, iteracoes=args.iteracoes)
    if args.velocidade != 1.0:
        import librosa
        y_lofi = librosa.effects.time_stretch(y_lofi, rate=args.velocidade)
    y_lofi = 0.95 * y_lofi / (np.abs(y_lofi).max() + 1e-8)
    sf.write(saida, y_lofi, audio.SR)
    print(f"gravado: {saida}")

    if args.referencia:
        y_ref = audio.mel_para_audio(db, fase_inicial=fase, iteracoes=args.iteracoes)
        ref = os.path.splitext(saida)[0] + "_referencia.wav"
        sf.write(ref, 0.95 * y_ref / (np.abs(y_ref).max() + 1e-8), audio.SR)
        print(f"gravado: {ref}")


if __name__ == "__main__":
    main()
