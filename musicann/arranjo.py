"""Seções da música e automações por seção.

Lo-fi não é um efeito fixo do começo ao fim: os produtores abrem e fecham o
filtro conforme a seção, tiram a bateria na introdução, jogam reverb na
última palavra antes de uma virada e fecham a música abafando até sumir.

detectar_secoes() acha os trechos da música (pela repetição do conteúdo
harmônico e do timbre) e dá um papel a cada um: introdução, verso, refrão
ou final. As funções curva_*() transformam esses papéis em curvas
amostra a amostra, que producao.produzir() aplica.
"""
import librosa
import numpy as np
from scipy import signal

PAPEIS = ("intro", "verso", "refrao", "final")


def detectar_secoes(mix, voz, sr):
    """Lista de dicts {ini, fim, papel} (segundos, linha do tempo original).

    1. Divide a música em trechos com agrupamento hierárquico sobre chroma e
       MFCC sincronizados às batidas (librosa.segment.agglomerative).
    2. Introdução: trechos do começo sem voz. Final: trechos do fim sem voz.
    3. No meio, os trechos mais "cheios" (energia acima do percentil 60) são
       tratados como refrão; os outros, como verso.
    """
    sr2 = 22050
    y = librosa.resample(mix, orig_sr=sr, target_sr=sr2)
    v = librosa.resample(voz, orig_sr=sr, target_sr=sr2)
    dur = len(y) / sr2
    _, batidas = librosa.beat.beat_track(y=y, sr=sr2)
    croma = librosa.feature.chroma_cqt(y=y, sr=sr2)
    mfcc = librosa.feature.mfcc(y=y, sr=sr2, n_mfcc=13)
    feat = np.vstack([croma, (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-6) / 3])
    feat = librosa.util.sync(feat, batidas, aggregate=np.median)
    k = int(np.clip(round(dur / 18), 4, 14))
    limites = librosa.segment.agglomerative(feat, min(k, feat.shape[1] - 1))
    t_bat = librosa.frames_to_time(librosa.util.fix_frames(batidas, x_min=0), sr=sr2)
    tempos = sorted({0.0, *[float(t_bat[min(i, len(t_bat) - 1)]) for i in limites if i > 0], dur})
    trechos = [(a, b) for a, b in zip(tempos[:-1], tempos[1:]) if b - a > 2]

    rms = lambda s, a, b: float(np.sqrt(np.mean(s[int(a * sr2) : int(b * sr2)] ** 2)) + 1e-9)
    e_voz = np.array([rms(v, a, b) for a, b in trechos])
    e_mix = np.array([rms(y, a, b) for a, b in trechos])
    com_voz = e_voz / e_voz.max() > 0.2
    papeis = ["verso"] * len(trechos)
    i = 0
    while i < len(trechos) and not com_voz[i] and trechos[i][1] < 0.3 * dur:
        papeis[i] = "intro"
        i += 1
    j = len(trechos) - 1
    while j > i and not com_voz[j] and trechos[j][0] > 0.6 * dur:
        papeis[j] = "final"
        j -= 1
    meio = [m for m in range(i, j + 1)]
    if meio:
        limiar = np.percentile(e_mix[meio], 60)
        for m in meio:
            if e_mix[m] >= limiar and com_voz[m]:
                papeis[m] = "refrao"

    secoes = []
    for (a, b), p in zip(trechos, papeis):
        if secoes and secoes[-1]["papel"] == p:
            secoes[-1]["fim"] = b
        else:
            secoes.append(dict(ini=a, fim=b, papel=p))
    return secoes


def alinhar_aos_compassos(secoes, compassos, duracao):
    """Leva cada fronteira de seção para o início de compasso mais próximo,
    se houver um a menos de meio compasso. Em trechos sem batida (voz
    sozinha, teclado sem bateria) a fronteira fica onde está; senão, numa
    música em que a bateria só entra depois de um minuto, todas iriam para
    o primeiro compasso e o começo viraria uma introdução só."""
    if len(compassos) == 0:
        return secoes
    meio = 0.5 * float(np.median(np.diff(compassos))) if len(compassos) > 1 else 1.0
    saida = []
    for s in secoes:
        perto = float(compassos[np.argmin(np.abs(compassos - s["ini"]))])
        ini = 0.0 if s["ini"] <= 0 else (perto if abs(perto - s["ini"]) <= meio else float(s["ini"]))
        saida.append(dict(s, ini=ini))
    for k in range(len(saida)):
        saida[k]["fim"] = saida[k + 1]["ini"] if k + 1 < len(saida) else duracao
    return [s for s in saida if s["fim"] - s["ini"] > 1]


def juntar_curtas(secoes, minimo):
    """Seções mais curtas que `minimo` segundos viram parte da anterior (ou
    da seguinte, se for a primeira): evita quebras e trocas de loop a cada
    3 s quando a detecção picota um trecho."""
    s = [dict(x) for x in secoes]
    mudou = True
    while mudou and len(s) > 1:
        mudou = False
        for k, x in enumerate(s):
            # O final fica mesmo curto (é onde a música se despede: bateria
            # saindo e filtro fechando).
            if k == len(s) - 1 and x["papel"] == "final" and x["fim"] - x["ini"] >= 4:
                continue
            if x["fim"] - x["ini"] < minimo:
                if k > 0:
                    s[k - 1]["fim"] = x["fim"]
                else:
                    s[1]["ini"] = x["ini"]
                del s[k]
                mudou = True
                break
    juntas = []
    for x in s:
        if juntas and juntas[-1]["papel"] == x["papel"]:
            juntas[-1]["fim"] = x["fim"]
        else:
            juntas.append(x)
    return juntas


def suavizar(curva, sr, ms=80):
    n = max(1, int(sr * ms / 1000))
    return signal.fftconvolve(curva, np.ones(n) / n, mode="same")


def curva_por_papel(n, sr, secoes, valores):
    """Curva amostra a amostra com o valor do papel de cada seção (papéis sem
    valor próprio usam o do verso)."""
    c = np.full(n, valores["verso"], dtype=np.float64)
    for s in secoes:
        c[int(s["ini"] * sr) : int(s["fim"] * sr)] = valores.get(s["papel"], valores["verso"])
    return c


# Tratamento de cada papel: refrão mais cheio que o verso, bateria e filtro
# marcando as mudanças, ponte e introdução mais vazias.
PERFIL = {
    #               bateria  corte bateria  base (x corte)  reverb voz
    "intro":        (0.0,    3000,          0.6,            0.25),
    "verso":        (1.0,    3500,          0.8,            0.10),
    "refrao":       (1.0,    8000,          1.35,           0.20),
    "ponte":        (0.0,    3000,          0.6,            0.35),
    "instrumental": (1.0,    6000,          1.2,            0.15),
    "final":        (1.0,    3500,          0.8,            0.25),
}


def perfil(papel):
    return PERFIL.get(papel, PERFIL["verso"])


def rampa(c, sr, ini, fim, de, para):
    a, b = int(ini * sr), min(len(c), int(fim * sr))
    if b > a:
        c[a:b] = np.linspace(de, para, b - a)


def passa_baixa_variavel(y, cortes, sr, bloco=1024):
    """Passa-baixa cujo corte muda ao longo do tempo (automação de filtro).
    Processa em blocos, levando o estado do filtro de um bloco para o outro."""
    saida = np.empty_like(y)
    zi = np.zeros((2, 2))
    for i in range(0, len(y), bloco):
        fc = float(np.clip(np.mean(cortes[i : i + bloco]), 80, sr / 2 - 200))
        sos = signal.butter(4, fc, "lowpass", fs=sr, output="sos")
        saida[i : i + bloco], zi = signal.sosfilt(sos, y[i : i + bloco], zi=zi)
    return saida


def fins_de_frase(voz, sr, antes_de, janela=6.0):
    """Para cada instante em `antes_de`, o fim da última frase cantada nos
    `janela` segundos anteriores (onde jogar o reverb), ou None."""
    hop = 512
    e = librosa.feature.rms(y=voz, hop_length=hop)[0]
    ativo = e > 0.15 * np.percentile(e, 95)
    t = librosa.frames_to_time(np.arange(len(e)), sr=sr, hop_length=hop)
    fins = t[1:][ativo[:-1] & ~ativo[1:]]
    saida = []
    for b in antes_de:
        cand = fins[(fins <= b + 0.3) & (fins >= b - janela)]
        saida.append(float(cand[-1]) if len(cand) else None)
    return saida
