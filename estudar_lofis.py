"""Estuda o que as lo-fi do YouTube mudam em relação à música original.

Para cada par de pares.csv que já tem alinhamento pela voz (dados/voz/,
de preparar_voz.py), separa a lo-fi e a original em voz, bateria, baixo e o
resto (Demucs) e compara parte a parte, no tempo alinhado:

  mesma_<parte>  correlação dos ataques da parte na lo-fi com os da mesma
                 parte na original (0 = outra gravação, ~0,7+ = a mesma
                 gravação, só tratada). Diz se o produtor manteve ou trocou
                 a bateria, o baixo e o resto (teclados, samples, guitarras).
  harmonia       correlação do chroma do resto (já corrigido o tom): os
                 acordes são os mesmos?
  <parte>_db     volume da parte em relação à voz, na lo-fi menos na original
  voz_brilho     quanto o centroide espectral da voz mudou (oitavas)
  chiado_db      piso de ruído da lo-fi nos trechos mais quietos, acima de 4 kHz
  velocidade, tom, cobertura (fração da lo-fi alinhada com a original)

As mesmas medidas, nas lo-fi geradas por este projeto (exemplos/completas/),
servem de referência: lá o instrumental é a própria gravação original.

    python estudar_lofis.py
"""
import csv
import json
import os

import librosa
import numpy as np

SR = 22050
HOP = 512
PASTA = os.path.join("dados", "estudo")


def stems(caminho, modelo, cache):
    """Stems mono a 22 kHz (float16), guardados em dados/estudo/."""
    if os.path.exists(cache):
        return {k: v.astype(np.float32) for k, v in np.load(cache).items()}
    import torch
    from separar import separar4
    y, _ = librosa.load(caminho, sr=44100, mono=False)
    partes = separar4(modelo, y, 44100, "cuda" if torch.cuda.is_available() else "cpu")
    partes = {k: librosa.resample(v, orig_sr=44100, target_sr=SR) for k, v in partes.items()}
    np.savez(cache, **{k: v.astype(np.float16) for k, v in partes.items()})
    return partes


def ataques(y):
    return librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)


def correlacao_ataques(a_lofi, a_orig, mapa, ok, janela=256, folga=2):
    """Mediana, em janelas de ~6 s, da correlação entre os ataques da lo-fi e
    os da original no ponto alinhado (com folga de ±2 quadros)."""
    from scipy.ndimage import maximum_filter1d
    o = maximum_filter1d(a_orig, 2 * folga + 1)[np.clip(mapa, 0, len(a_orig) - 1)]
    n = min(len(a_lofi), len(o))
    r = []
    for s in range(0, n - janela, janela // 2):
        sel = ok[s : s + janela]
        if sel.mean() < 0.8:
            continue
        x, y = a_lofi[s : s + janela], o[s : s + janela]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r.append(np.corrcoef(x, y)[0, 1])
    return float(np.median(r)) if r else np.nan


def rms_db(y):
    return float(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-9))


def comparar(L, O, mapa, ok, tom):
    m = {}
    for p in ("drums", "bass", "other", "vocals"):
        m["mesma_" + p] = correlacao_ataques(ataques(L[p]), ataques(O[p]), mapa, ok)
        if p != "vocals":
            m[p + "_db"] = (rms_db(L[p]) - rms_db(L["vocals"])) - (rms_db(O[p]) - rms_db(O["vocals"]))
    cl = librosa.feature.chroma_cqt(y=L["other"], sr=SR, hop_length=HOP)
    co = np.roll(librosa.feature.chroma_cqt(y=O["other"], sr=SR, hop_length=HOP), int(round(tom)), axis=0)
    co = co[:, np.clip(mapa, 0, co.shape[1] - 1)]
    n = min(cl.shape[1], co.shape[1])
    cos = (cl[:, :n] * co[:, :n]).sum(0) / (np.linalg.norm(cl[:, :n], axis=0) * np.linalg.norm(co[:, :n], axis=0) + 1e-9)
    m["harmonia"] = float(np.median(cos[ok[:n]]))
    cen = lambda v: np.median(librosa.feature.spectral_centroid(y=v, sr=SR, hop_length=HOP)[0][librosa.feature.rms(y=v, hop_length=HOP)[0] > 0.02])
    m["voz_brilho"] = float(np.log2(cen(L["vocals"]) / cen(O["vocals"])))
    mix = sum(L.values())
    S = np.abs(librosa.stft(mix, n_fft=2048, hop_length=HOP)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    agudo, total = S[f > 4000].sum(0), S.sum(0)
    quietos = total < np.percentile(total, 10)
    m["chiado_db"] = float(10 * np.log10(np.median(agudo[quietos]) / np.median(total) + 1e-12))
    return m


def main():
    os.makedirs(PASTA, exist_ok=True)
    from demucs.pretrained import get_model
    import torch
    modelo = get_model("htdemucs").to("cuda" if torch.cuda.is_available() else "cpu").eval()
    pares = [r["nome"] for r in csv.DictReader(open("pares.csv", encoding="utf-8"), delimiter=";")]
    resultados = {}
    # Lo-fi reais do YouTube.
    for n in pares:
        arq_voz = os.path.join("dados", "voz", n + ".npz")
        if not (os.path.exists(arq_voz) and os.path.exists(os.path.join("Musics", "Lofi", n + ".mp3"))):
            continue
        par = np.load(arq_voz)
        ok = np.isfinite(par["custo"]) & (par["custo_fino"] <= 0.4)
        if ok.mean() < 0.3:
            continue
        L = stems(os.path.join("Musics", "Lofi", n + ".mp3"), modelo, os.path.join(PASTA, f"lofi_{n}.npz"))
        O = stems(os.path.join("Musics", "Normal", n + ".mp3"), modelo, os.path.join(PASTA, f"orig_{n}.npz"))
        mapa = par["mapa"]
        n_l = min(len(mapa), len(ataques(L["vocals"])))
        m = comparar(L, O, mapa[:n_l], ok[:n_l], float(par["tom"]))
        m.update(velocidade=float(np.median(par["inclinacao"][ok])), tom=float(par["tom"]), cobertura=float(ok.mean()))
        resultados[n] = m
        print(n, json.dumps({k: round(v, 2) for k, v in m.items()}), flush=True)
    # Lo-fi deste projeto (referência: instrumental da própria gravação).
    for arq in sorted(os.listdir(os.path.join("exemplos", "completas"))):
        n, ext = os.path.splitext(arq)
        if ext != ".wav" or not os.path.exists(os.path.join("Musics", "Normal", n + ".mp3")):
            continue
        L = stems(os.path.join("exemplos", "completas", arq), modelo, os.path.join(PASTA, f"nossa_{n}.npz"))
        O = stems(os.path.join("Musics", "Normal", n + ".mp3"), modelo, os.path.join(PASTA, f"orig_{n}.npz"))
        n_l = len(ataques(L["vocals"]))
        mapa = np.round(np.arange(n_l) * 0.97).astype(int)
        m = comparar(L, O, mapa, np.ones(n_l, bool), -0.5)
        m.update(velocidade=0.97, tom=-0.5, cobertura=1.0)
        resultados["(nossa) " + n] = m
        print("(nossa)", n, json.dumps({k: round(v, 2) for k, v in m.items()}), flush=True)
    json.dump(resultados, open(os.path.join(PASTA, "resultados.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
