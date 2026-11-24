"""Áudio <-> espectrograma de Mel em escala log, e volta para áudio."""
import numpy as np
import librosa

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128

# Faixa usada para levar o log-Mel (em dB) para perto de [-1, 1].
DB_MIN = -100.0
DB_MAX = 20.0


def carregar(caminho, sr=SR):
    y, _ = librosa.load(caminho, sr=sr, mono=True)
    return y


def stft(y):
    return librosa.stft(y, n_fft=N_FFT, hop_length=HOP)


def mel_db(y):
    """Espectrograma de Mel em dB, formato (N_MELS, quadros)."""
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)


def normalizar(db):
    db = np.clip(db, DB_MIN, DB_MAX)
    return (2 * (db - DB_MIN) / (DB_MAX - DB_MIN) - 1).astype(np.float32)


def desnormalizar(x):
    return (x + 1) / 2 * (DB_MAX - DB_MIN) + DB_MIN


def aplicar_diferenca(y, db_entrada, db_saida, limite=24.0):
    """Leva y de db_entrada para db_saida mexendo só no ganho de cada faixa.

    Em vez de reconstruir o áudio do zero a partir do Mel (que perde fase e
    resolução), calculo quanto cada banda subiu ou desceu, espalho essa
    diferença pelas frequências do STFT e aplico no STFT da própria entrada,
    mantendo a fase original. O som fica bem mais limpo; o preço é que só dá
    para mudar o que já existe na entrada (não cria sons novos).
    """
    S = stft(y)
    n = min(S.shape[1], db_entrada.shape[1])
    delta = np.clip(db_saida[:, :n] - db_entrada[:, :n], -limite, limite)
    base = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    pesos = base / (base.sum(axis=0, keepdims=True) + 1e-8)  # (mels, bins), soma 1 por bin
    ganho_db = pesos.T @ delta  # (bins, quadros)
    S = S[:, :n] * librosa.db_to_amplitude(ganho_db)
    return librosa.istft(S, hop_length=HOP, length=(n - 1) * HOP)


def mel_para_audio(db, fase_inicial=None, iteracoes=64):
    """Converte log-Mel (dB) em áudio.

    O Mel descarta a fase e boa parte da resolução em frequência. Primeiro
    recupero a magnitude linear (mínimos quadrados não negativos) e depois
    estimo a fase com Griffin-Lim. Se houver uma fase de partida (por exemplo,
    a da música original), o resultado fica bem mais limpo do que partir de
    uma fase aleatória.
    """
    S = librosa.db_to_power(db)
    mag = librosa.feature.inverse.mel_to_stft(S, sr=SR, n_fft=N_FFT, power=2.0)
    if fase_inicial is None:
        fase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    else:
        n = min(mag.shape[1], fase_inicial.shape[1])
        mag = mag[:, :n]
        fase = np.exp(1j * np.angle(fase_inicial[:, :n]))
    comprimento = (mag.shape[1] - 1) * HOP
    for _ in range(iteracoes):
        y = librosa.istft(mag * fase, hop_length=HOP, length=comprimento)
        fase = np.exp(1j * np.angle(stft(y)[:, : mag.shape[1]]))
    return librosa.istft(mag * fase, hop_length=HOP, length=comprimento)
