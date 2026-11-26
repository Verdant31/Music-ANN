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


def separar_assobio(voz, sr, fmin=1000.0, fmax=3000.0, concentracao=0.6, minimo_s=0.15):
    """Tira da voz separada um assobio (ou flauta, apito): o instrumento que o
    Demucs pôs na voz.

    Em Chill Bill o assobio do gancho saiu inteiro na faixa da voz (a
    introdução e o final são só ele, a ~1,6 kHz). Como voz, ele passava
    limpo, no volume da voz e sem nada de lo-fi, e a música soava como a
    original com outra base. Aqui: onde a voz é um tom quase puro e agudo
    (pitch entre `fmin` e `fmax`, com >= `concentracao` da energia de 800 Hz
    a 8 kHz nos 3 primeiros harmônicos), por pelo menos `minimo_s`
    segundos, as faixas de frequência desses harmônicos saem da voz. Voz
    cantada ou falada tem a energia espalhada em muitos harmônicos (0,06-0,08
    nas músicas de rap; o assobio, 0,85).
    voz: (2, n). Devolve (voz sem o assobio, assobio), os dois (2, n)."""
    sr2, hop2, n_fft = 22050, 256, 4096
    y = librosa.resample(voz.mean(0), orig_sr=sr, target_sr=sr2)
    f0, vz, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr2, frame_length=2048, hop_length=hop2)
    P = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop2)) ** 2
    f2 = librosa.fft_frequencies(sr=sr2, n_fft=2048)
    banda = (f2 > 800) & (f2 < 8000)
    m = min(P.shape[1], len(f0))
    e = np.sqrt(P[:, :m].mean(0))
    assobio = np.zeros(m, bool)
    for t in np.flatnonzero(vz[:m] & np.isfinite(f0[:m]) & (e > 0.1 * np.percentile(e, 90))):
        perto = np.zeros(len(f2), bool)
        for h in (1, 2, 3):
            perto |= np.abs(1200 * np.log2((f2 + 1e-9) / (h * f0[t]))) < 70
        assobio[t] = P[perto & banda, t].sum() > concentracao * P[banda, t].sum()
    # Só trechos que duram (tira detecções soltas de uma vogal aguda).
    rot, k = ndimage.label(assobio)
    for i in range(1, k + 1):
        if (rot == i).sum() * hop2 / sr2 < minimo_s:
            assobio[rot == i] = False
    if not assobio.any():
        return voz, np.zeros_like(voz)
    # Máscara nos harmônicos, na resolução do áudio original (mesmo passo no tempo).
    hop = int(round(hop2 * sr / sr2))
    X = librosa.stft(voz, n_fft=n_fft, hop_length=hop)
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    M = np.zeros(X.shape[1:])
    for t in np.flatnonzero(assobio):
        if t >= M.shape[1]:
            break
        for h in (1, 2, 3, 4):
            M[np.abs(1200 * np.log2((f + 1e-9) / (h * f0[t]))) < 70, t] = 1
    M = ndimage.uniform_filter(M, size=(3, 5))
    so_assobio = librosa.istft(X * M, hop_length=hop, length=voz.shape[1])
    return voz - so_assobio, so_assobio
