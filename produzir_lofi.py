"""Gera a versão lo-fi de uma música montando-a como um produtor
(ver musicann/producao.py).

    python produzir_lofi.py musica.mp3
    python produzir_lofi.py musica.mp3 --inicio 60 --duracao 30 --velocidade 0.88
"""
import argparse
import os

import librosa
import soundfile as sf

from musicann import producao


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--saida")
    ap.add_argument("--inicio", type=float, default=0.0)
    ap.add_argument("--duracao", type=float, default=None)
    ap.add_argument("--velocidade", type=float, default=None)
    ap.add_argument("--corte", type=float, default=None, help="Hz do passa-baixa da base")
    ap.add_argument("--semente", type=int, default=0)
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from separar import separar4
    y, _ = librosa.load(args.entrada, sr=producao.SR, mono=True, offset=args.inicio, duration=args.duracao)
    modelo = get_model("htdemucs").to("cuda").eval()
    partes = separar4(modelo, y, producao.SR, "cuda")

    ajustes = producao.Ajustes()
    if args.velocidade:
        ajustes.velocidade = args.velocidade
    if args.corte:
        ajustes.corte_base = args.corte
    mix, info = producao.produzir(partes, ajustes, semente=args.semente)
    saida = args.saida or os.path.splitext(os.path.basename(args.entrada))[0] + "_lofi.wav"
    sf.write(saida, mix, producao.SR)
    print(f"gravado: {saida} | {info}")


if __name__ == "__main__":
    main()
