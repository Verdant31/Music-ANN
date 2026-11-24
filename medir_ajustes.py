"""Mede, em cada par real, os ajustes que a lo-fi do YouTube usou.

Para cada música (usando o alinhamento pela voz de preparar_voz.py):
  velocidade  quanto a lo-fi é mais lenta (inclinação do alinhamento)
  tom         semitons de diferença (negativo = mais grave)
  corte       Hz onde o instrumental da lo-fi já perdeu 6 dB em relação à
              original (é exatamente o "corte" do passa-baixa da montagem)
  graves_db   quanto os graves (< 120 Hz) mudaram em relação aos médios
  reverb      quanto de reverb a voz ganhou: mede a "cauda" (energia da voz
              lo-fi nos momentos em que a voz original está em silêncio) e
              converte para o parâmetro da montagem aplicando o próprio
              reverb da montagem em vozes de exemplo (calibração)

E guarda também as características de cada original, que são a entrada da
rede (musicann/ajustes.py). Grava dados/ajustes.npz.

    python medir_ajustes.py
"""
import os

import librosa
import numpy as np

from musicann import audio, producao
from musicann.dados import carregar_pares

FREQS = librosa.mel_frequencies(n_mels=audio.N_MELS, fmax=audio.SR / 2)
ALVOS = ["velocidade", "tom", "corte", "graves_db", "reverb"]


def perfil(par, ok):
    """Diferença média lo-fi - original (dB) por banda, nos quadros alinhados."""
    o = par["original"][:, np.clip(par["mapa"][ok], 0, par["original"].shape[1] - 1)]
    l = par["lofi"][:, ok]
    d = l.mean(1) - o.mean(1)
    medios = (FREQS > 200) & (FREQS < 1500)
    return d - d[medios].mean()


def corte_de(dif):
    """Maior frequência a partir da qual a lo-fi ficou 6 dB abaixo."""
    acima = FREQS > 1000
    abaixo_6 = acima & (dif < -6)
    if not abaixo_6.any():
        return float(FREQS[-1])
    return float(FREQS[np.flatnonzero(abaixo_6)[0]])


def cauda(voz_o, voz_l, ok, mapa):
    """Energia da voz lo-fi (dB) nos quadros em que a voz original está em
    silêncio, relativa aos quadros com voz. Mais reverb = cauda mais alta."""
    e_o = voz_o.max(0)[np.clip(mapa[ok], 0, voz_o.shape[1] - 1)]
    e_l = voz_l.max(0)[ok]
    silencio = e_o < np.percentile(e_o, 15)
    cantando = e_o > np.percentile(e_o, 60)
    if silencio.sum() < 20 or cantando.sum() < 20:
        return np.nan
    return float(np.median(e_l[silencio]) - np.median(e_l[cantando]))


def calibrar_reverb(vozes):
    """Mede a cauda que o reverb da montagem produz para cada valor de mix."""
    mixes = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    caudas = []
    for m in mixes:
        c = []
        for v in vozes:
            molhada = producao.reverb(librosa.resample(v, orig_sr=audio.SR, target_sr=producao.SR), m)
            seco, mol = audio.mel_db(v), audio.mel_db(librosa.resample(molhada, orig_sr=producao.SR, target_sr=audio.SR))
            n = min(seco.shape[1], mol.shape[1])
            c.append(cauda(seco[:, :n], mol[:, :n], np.ones(n, bool), np.arange(n)))
        caudas.append(np.nanmedian(c))
    return mixes, np.maximum.accumulate(np.array(caudas))  # monótona para poder inverter


def caracteristicas(m_db, y):
    """Entrada da rede: como a música original soa (espectro médio e variação
    por faixa, em 32 grupos de bandas), andamento e brilho."""
    g = m_db.reshape(32, 4, -1)
    media = g.mean((1, 2))
    var = g.std(2).mean(1)
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=audio.SR)[0])[0])
    centroide = float(librosa.feature.spectral_centroid(y=y, sr=audio.SR).mean())
    return np.concatenate([media - media.mean(), var, [tempo / 100, centroide / 2000]]).astype(np.float32)


def main():
    mix, inst, voz = (carregar_pares(f"dados/{p}") for p in ("mix", "inst", "voz"))
    nomes = sorted(set(mix) & set(inst) & set(voz))

    exemplos = [audio.carregar(os.path.join("Musics", "Separado", "Normal", f"{n}_voz.wav"))[30 * audio.SR : 45 * audio.SR]
                for n in nomes[:: max(1, len(nomes) // 6)]]
    mixes, caudas = calibrar_reverb(exemplos)
    print("calibração do reverb (mix -> cauda dB):", dict(zip(mixes.round(2), caudas.round(1))))

    X, Y, usados = [], [], []
    for n in nomes:
        ok = np.isfinite(voz[n]["custo"]) & (voz[n]["custo_fino"] <= 0.4)
        if ok.mean() < 0.3:
            continue
        dif = perfil(inst[n], ok)
        c = cauda(voz[n]["original"], voz[n]["lofi"], ok, voz[n]["mapa"])
        if not np.isfinite(c):
            continue
        reverb = float(np.interp(c, caudas, mixes))
        alvo = [float(np.median(voz[n]["inclinacao"][ok])), float(voz[n]["tom"]), corte_de(dif),
                float(dif[FREQS < 120].mean()), reverb]
        y = audio.carregar(os.path.join("Musics", "Normal", n + ".mp3"))
        X.append(caracteristicas(audio.mel_db(y), y))
        Y.append(alvo)
        usados.append(n)
        print(f"{n:32} " + " ".join(f"{k} {v:7.2f}" for k, v in zip(ALVOS, alvo)), flush=True)

    np.savez("dados/ajustes.npz", X=np.array(X), Y=np.array(Y, np.float32), nomes=np.array(usados), alvos=np.array(ALVOS))
    print(f"\n{len(usados)} músicas | médias: " + " ".join(f"{k} {v:.2f}" for k, v in zip(ALVOS, np.mean(Y, 0))))


if __name__ == "__main__":
    main()
