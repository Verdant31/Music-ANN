"""Substituição de bateria (drum replacement) com som lo-fi.

Em vez de colar um loop de fora (que precisa ser alinhado com a música e,
em gravações tocadas ao vivo ou antigas, às vezes não alinha de jeito
nenhum), detecto cada golpe da bateria original, classifico em bumbo, caixa
ou chimbal pelo timbre, e disparo no mesmo instante uma amostra lo-fi do
mesmo tipo, recortada de um loop do Lo-Fi Drums Dataset. O tempo vem do
próprio baterista da gravação: não tem como descompassar.
"""
import librosa
import numpy as np

SR = 44100
DURACAO = {"bumbo": 0.45, "caixa": 0.35, "chimbal": 0.12}


def _faixas(y, t, sr=SR, janela=0.05):
    """Energia relativa de grave (< 150 Hz), médio (150 Hz-4 kHz) e agudo
    (> 6 kHz) nos primeiros 50 ms de cada golpe."""
    saida = []
    for x in t:
        a = int(x * sr)
        trecho = y[a : a + int(janela * sr)]
        if len(trecho) < 256:
            saida.append((0, 0, 0))
            continue
        S = np.abs(np.fft.rfft(trecho * np.hanning(len(trecho)))) ** 2
        f = np.fft.rfftfreq(len(trecho), 1 / sr)
        tot = S.sum() + 1e-12
        saida.append((S[f < 150].sum() / tot, S[(f >= 150) & (f < 4000)].sum() / tot, S[f >= 6000].sum() / tot))
    return np.array(saida)


def classificar(y, sr=SR, limiar=0.15):
    """Golpes da bateria: lista de (instante s, tipo, força 0-1)."""
    env = librosa.onset.onset_strength(y=y, sr=sr)
    q = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="frames", backtrack=True)
    if len(q) == 0:
        return []
    forca = env[np.minimum(q + 2, len(env) - 1)]
    forca = forca / (np.percentile(forca, 95) + 1e-9)
    t = librosa.frames_to_time(q, sr=sr)
    bandas = _faixas(y, t, sr)
    golpes = []
    for ti, fo, (g, m, a) in zip(t, forca, bandas):
        if fo < limiar:  # ataques fracos: em gravações antigas, vazamento de outros instrumentos
            continue
        tipo = "bumbo" if g > 0.45 else ("chimbal" if a > 0.35 and g < 0.15 else "caixa")
        golpes.append((float(ti), tipo, float(min(fo, 1.5))))
    return golpes


def amostras_do_loop(loop, sr=SR):
    """Uma amostra de cada tipo (bumbo, caixa, chimbal) recortada do loop: o
    golpe mais forte de cada tipo. Devolve {tipo: áudio estéreo (2, n)}."""
    if loop.ndim == 1:
        loop = np.stack([loop, loop])
    golpes = classificar(loop.mean(0), sr, limiar=0.05)
    amostras = {}
    for tipo in DURACAO:
        do_tipo = [g for g in golpes if g[1] == tipo]
        if not do_tipo:
            continue
        t, _, _ = max(do_tipo, key=lambda g: g[2])
        a, b = int(t * sr), int((t + DURACAO[tipo]) * sr)
        s = loop[:, a:b].copy()
        r = min(int(0.03 * sr), s.shape[1] // 3)
        s[:, -r:] *= np.linspace(1, 0, r)
        amostras[tipo] = s / (np.abs(s).max() + 1e-9)
    # Sem chimbal no loop, usa a ponta aguda da caixa; sem caixa, nada a fazer.
    if "chimbal" not in amostras and "caixa" in amostras:
        amostras["chimbal"] = 0.4 * amostras["caixa"][:, : int(0.08 * sr)]
    return amostras


def substituir(n, golpes, amostras, sr=SR):
    """Bateria nova (2, n): cada golpe original vira a amostra lo-fi do mesmo
    tipo, no mesmo instante, com volume pela força do golpe (comprimida, para
    não variar demais)."""
    saida = np.zeros((2, n))
    for t, tipo, forca in golpes:
        s = amostras.get(tipo)
        if s is None:
            continue
        a = int(t * sr)
        if a >= n:
            continue
        m = min(s.shape[1], n - a)
        saida[:, a : a + m] += (forca ** 0.6) * s[:, :m]
    return saida
