"""Separa voz e instrumental de cada música com o Demucs (htdemucs).

Grava Musics/Separado/<Normal|Lofi>/<nome>_voz.wav e <nome>_inst.wav, a
22.050 Hz mono, que é a taxa do resto do projeto.

    python separar.py
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from musicann import audio


def separar(modelo, caminho, dev):
    y, _ = librosa.load(caminho, sr=modelo.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    x = torch.from_numpy(y).float()
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        fontes = apply_model(modelo, x[None].to(dev), split=True, overlap=0.25, progress=False)[0]
    fontes = fontes * (ref.std() + 1e-8) + ref.mean()
    i_voz = modelo.sources.index("vocals")
    voz = fontes[i_voz].mean(0).cpu().numpy()
    inst = (fontes.sum(0) - fontes[i_voz]).mean(0).cpu().numpy()
    para = lambda s: librosa.resample(s, orig_sr=modelo.samplerate, target_sr=audio.SR).astype(np.float32)
    return para(voz), para(inst)


def separar4(modelo, y, sr_entrada, dev, estereo=False):
    """Separa em voz, bateria, baixo e o resto. Devolve um dict de áudios na
    taxa do modelo (44,1 kHz): mono, ou (2, n) com estereo=True."""
    if sr_entrada != modelo.samplerate:
        y = librosa.resample(y, orig_sr=sr_entrada, target_sr=modelo.samplerate)
    x = torch.from_numpy(np.stack([y, y]) if y.ndim == 1 else y).float()
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        fontes = apply_model(modelo, x[None].to(dev), split=True, overlap=0.25, progress=False)[0]
    fontes = fontes * (ref.std() + 1e-8) + ref.mean()
    if estereo:
        return {nome: fontes[i].cpu().numpy() for i, nome in enumerate(modelo.sources)}
    return {nome: fontes[i].mean(0).cpu().numpy() for i, nome in enumerate(modelo.sources)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musicas", default="Musics")
    args = ap.parse_args()
    dev = "cuda"
    modelo = get_model("htdemucs").to(dev).eval()
    for lado in ["Normal", "Lofi"]:
        pasta = os.path.join(args.musicas, lado)
        saida = os.path.join(args.musicas, "Separado", lado)
        os.makedirs(saida, exist_ok=True)
        for f in sorted(os.listdir(pasta)):
            if not f.endswith(".mp3"):
                continue
            nome = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(saida, nome + "_inst.wav")):
                continue
            voz, inst = separar(modelo, os.path.join(pasta, f), dev)
            sf.write(os.path.join(saida, nome + "_voz.wav"), voz, audio.SR)
            sf.write(os.path.join(saida, nome + "_inst.wav"), inst, audio.SR)
            print(lado, nome, flush=True)


if __name__ == "__main__":
    main()
