"""Conserto dos "buracos" que a separação deixa na voz e no resto.

Em cada golpe forte da bateria (bumbo, caixa), o Demucs dá para a bateria
quase toda a energia daquele instante: a voz e o resto ficam com um buraco
de ~100 ms, ~7 dB mais fraco que o normal (em Neighbors, medido no sinal: a
voz cai 11-12 dB em volta dos golpes fortes contra 4-5 dB num instante
qualquer; o resto cai 4 dB contra 0,5). Na música original
isso não se ouve, porque o próprio golpe tapa o buraco. Na lo-fi, a bateria
original sai e o loop não bate exatamente nos mesmos lugares (em meio-tempo,
bate na metade deles): o buraco fica exposto e a voz parece "falhar" por um
instante.

tapar_buracos() mede o nível de cada faixa de frequência logo antes e logo
depois de cada golpe e sobe o trecho do golpe até a média (geométrica) dos
dois, no máximo `max_db`. Se o cantor estava em pausa antes ou depois, não
mexe. Tapa uns 60% do buraco: o que falta é energia que foi inteira para a
bateria e não dá para recuperar subindo o volume do que sobrou.
"""
import librosa
import numpy as np
from scipy import ndimage

N_FFT, HOP = 1024, 128  # janela de 23 ms: o buraco tem ~60 ms


def _faixa_de_cada_bin(sr, n_faixas=24, fmin=60.0):
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    bordas = np.geomspace(fmin, sr / 2, n_faixas + 1)
    return np.clip(np.searchsorted(bordas, f) - 1, 0, n_faixas - 1)


def tapar_buracos(y, golpes, sr, max_db=12.0):
    """y: áudio (n,) ou (2, n); golpes: instantes (s) dos golpes fortes da
    bateria original. Devolve o áudio com os buracos tapados."""
    canais = y if y.ndim == 2 else y[None]
    faixa = _faixa_de_cada_bin(sr)
    nf = faixa.max() + 1
    mag = np.abs(librosa.stft(canais.mean(0), n_fft=N_FFT, hop_length=HOP))
    E = np.stack([np.sqrt((mag[faixa == b] ** 2).mean(0)) if np.any(faixa == b) else np.zeros(mag.shape[1])
                  for b in range(nf)])
    T = E.shape[1]
    # Só onde a parte está tocando: nível total acima da mediana da música e,
    # em cada faixa, acima do piso de ruído (percentil 20 da faixa).
    total = np.sqrt((E**2).sum(0))
    tocando = np.percentile(total, 50)
    piso = np.percentile(E, 20, axis=1)
    ganho = np.ones((nf, T))
    teto = 10 ** (max_db / 20)
    for g in golpes:
        k = int(round(g * sr / HOP))
        # antes: -120 a -70 ms; buraco: -60 a +60 ms; depois: +65 a +110 ms
        # (medido em Neighbors: a queda começa ~50 ms antes do golpe
        # detectado e volta ao normal ~60 ms depois)
        q = lambda ms: k + int(round(ms / 1000 * sr / HOP))
        a0, a1, h0, h1, d0, d1 = q(-120), q(-70), q(-60), q(60), q(65), q(110)
        if a0 < 0 or d1 >= T:
            continue
        antes, depois = E[:, a0:a1].mean(1), E[:, d0:d1].mean(1)
        if min(np.sqrt((antes**2).sum()), np.sqrt((depois**2).sum())) < tocando:
            continue  # pausa antes ou depois: o "buraco" pode ser o fim de uma frase
        ref = np.sqrt(antes * depois)
        g_ = np.clip(ref[:, None] / (E[:, h0:h1] + 1e-9), 1, teto)
        g_[ref <= piso] = 1
        ganho[:, h0:h1] = np.maximum(ganho[:, h0:h1], g_)
    ganho = ndimage.uniform_filter1d(ganho, 5, axis=1)
    ganho = ndimage.uniform_filter1d(ganho, 3, axis=0)
    G = ganho[faixa]
    saida = [librosa.istft(librosa.stft(c, n_fft=N_FFT, hop_length=HOP) * G, hop_length=HOP, length=canais.shape[1])
             for c in canais]
    saida = np.stack(saida).astype(y.dtype)
    return saida if y.ndim == 2 else saida[0]
