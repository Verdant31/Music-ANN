"""Gera a versão lo-fi de uma música montando-a como um produtor
(ver musicann/producao.py).

    python produzir_lofi.py musica.mp3
    python produzir_lofi.py musica.mp3 --inicio 60 --duracao 30 --velocidade 0.88
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf

from musicann import producao


def secoes_da_musica(caminho, y, inicio, duracao, pasta="dados/secoes"):
    """Seções do SongFormer, guardadas em dados/secoes/ (em músicas longas a
    análise leva minutos numa GPU de 12 GB; assim é feita uma vez só)."""
    import json
    os.makedirs(pasta, exist_ok=True)
    chave = f"{os.path.splitext(os.path.basename(caminho))[0]}_{inicio:g}_{duracao or 0:g}.json"
    arq = os.path.join(pasta, chave)
    if os.path.exists(arq):
        return json.load(open(arq, encoding="utf-8"))
    from musicann import estrutura
    s = estrutura.secoes(y, producao.SR)
    json.dump(s, open(arq, "w", encoding="utf-8"), indent=1)
    return s


def stems_da_musica(caminho, y, inicio, duracao, modelo_nome, pasta="dados/stems"):
    """Voz, bateria, baixo e o resto (estéreo), guardados em dados/stems/: a
    separação é a parte mais pesada e não muda entre uma versão e outra."""
    os.makedirs(pasta, exist_ok=True)
    nome = os.path.splitext(os.path.basename(caminho))[0]
    sufixo = "" if not inicio and not duracao else f"_{inicio:g}_{duracao or 0:g}"
    arq = os.path.join(pasta, f"{nome}{sufixo}_{modelo_nome}.npz")
    if os.path.exists(arq):
        return {k: v for k, v in np.load(arq).items()}
    import torch
    from demucs.pretrained import get_model
    from separar import separar4
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = get_model(modelo_nome).to(dev).eval()
    partes = separar4(modelo, y, producao.SR, dev, estereo=True)
    np.savez(arq, **{k: v.astype(np.float32) for k, v in partes.items()})
    return partes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entrada")
    ap.add_argument("--saida")
    ap.add_argument("--inicio", type=float, default=0.0)
    ap.add_argument("--duracao", type=float, default=None)
    ap.add_argument("--velocidade", type=float, default=None)
    ap.add_argument("--corte", type=float, default=None, help="Hz do passa-baixa da base")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--sem-songformer", action="store_true",
                    help="usa a detecção de seções simples em vez do SongFormer")
    ap.add_argument("--modelo", default="htdemucs", help="modelo do Demucs (htdemucs ou htdemucs_ft)")
    ap.add_argument("--bateria", default="auto", choices=["auto", "loop", "substituicao"],
                    help="loop lo-fi encaixado ou substituição golpe a golpe (auto: escolhe pela música)")
    args = ap.parse_args()

    y, _ = librosa.load(args.entrada, sr=producao.SR, mono=False, offset=args.inicio, duration=args.duracao)
    partes = stems_da_musica(args.entrada, y, args.inicio, args.duracao, args.modelo)

    ajustes = producao.Ajustes()
    if args.velocidade:
        ajustes.velocidade = args.velocidade
    if args.corte:
        ajustes.corte_base = args.corte
    ajustes.bateria = args.bateria
    secoes = None
    if not args.sem_songformer:
        secoes = secoes_da_musica(args.entrada, y, args.inicio, args.duracao)
    mix, info = producao.produzir(partes, ajustes, semente=args.semente, secoes_externas=secoes)
    saida = args.saida or os.path.splitext(os.path.basename(args.entrada))[0] + "_lofi.wav"
    sf.write(saida, mix, producao.SR)
    # Seções que a montagem usou de fato (no tempo da lo-fi), para a página.
    import json
    json.dump(info["secoes"], open(os.path.splitext(saida)[0] + ".secoes.json", "w", encoding="utf-8"))
    # Velocidade e demais escolhas, para a página converter o tempo de cada versão.
    json.dump(dict(info, velocidade=ajustes.velocidade), open(os.path.splitext(saida)[0] + ".info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"gravado: {saida} | {info}")


if __name__ == "__main__":
    main()
