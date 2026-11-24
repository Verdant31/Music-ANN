"""Cadeia de efeitos que imita uma produção lo-fi.

Serve de teste de controle: aplicada na própria original, gera pares
perfeitamente alinhados. Se a U-Net aprende bem esses pares, a rede funciona
e qualquer problema com os pares reais está nos dados.
"""
import numpy as np
from scipy import signal

from .audio import SR


def aplicar(y, rng):
    p = dict(
        corte=rng.uniform(3000, 5500),     # Hz, passa-baixa (some o brilho)
        grave=rng.uniform(80, 150),        # Hz, passa-alta leve
        drive=rng.uniform(1.5, 3.0),       # saturação de fita
        wow=rng.uniform(0.0008, 0.002),    # oscilação lenta de velocidade
        chiado=rng.uniform(0.004, 0.01),   # ruído de fundo
        estalos=rng.uniform(2, 6),         # estalos de vinil por segundo
    )
    x = y / (np.abs(y).max() + 1e-8)

    # Wow: lê o áudio com um atraso que oscila devagar (~0,5 Hz).
    t = np.arange(len(x))
    atraso = p["wow"] * SR * np.sin(2 * np.pi * 0.5 * t / SR)
    x = np.interp(t + atraso, t, x)

    sos = signal.butter(4, p["corte"], "lowpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)
    sos = signal.butter(2, p["grave"], "highpass", fs=SR, output="sos")
    x = signal.sosfilt(sos, x)

    x = np.tanh(p["drive"] * x) / np.tanh(p["drive"])

    # Chiado rosa e estalos de vinil.
    ruido = rng.standard_normal(len(x))
    ruido = signal.lfilter([0.049922, -0.095993, 0.050613, -0.004408], [1, -2.494956, 2.017265, -0.522189], ruido)
    x = x + p["chiado"] * ruido / (np.abs(ruido).max() + 1e-8)
    n_estalos = int(p["estalos"] * len(x) / SR)
    pos = rng.integers(0, len(x) - 64, n_estalos)
    for i in pos:
        x[i : i + 64] += rng.uniform(0.05, 0.2) * np.exp(-np.arange(64) / 8) * rng.choice([-1, 1])
    return (0.9 * x / (np.abs(x).max() + 1e-8)).astype(np.float32), p
