"""Monta a versão lo-fi de uma música do jeito que um produtor faria.

A música é separada em voz, bateria, baixo e o resto (Demucs). Depois:
  - voz: fica limpa, com um reverb leve
  - bateria: a original sai e entra um loop de bateria lo-fi de verdade,
    encaixado compasso a compasso nas batidas da música
  - baixo e o resto: corte de agudos, saturação de fita, oscilação lenta
    (wow/flutter) e o volume "respirando" com o bumbo (sidechain)
  - tudo desacelerado como um disco mais lento, com chiado de vinil por cima

Os ajustes ficam em `Ajustes`; a rede de `musicann/ajustes.py` aprende a
escolhê-los a partir dos pares reais.
"""
import glob
import os
import re
from dataclasses import dataclass

import librosa
import numpy as np
from scipy import signal

SR = 44100


@dataclass
class Ajustes:
    velocidade: float = 0.95       # 1 = sem desacelerar; 0.95 = 5% mais lento
    manter_tom: bool = True        # True: desacelera sem engrossar a voz; False: como disco (mais grave)
    corte_base: float = 4500.0     # Hz, passa-baixa no baixo e no resto (o "abafado")
    corte_final: float = 11000.0   # Hz, passa-baixa suave na mixagem toda
    corte_bateria: float = 7000.0  # Hz, bateria um pouco abafada para casar com a base
    corte_graves: float = 120.0    # Hz, abaixo disso a mixagem perde um pouco (menos "grossa")
    graves_db: float = -2.0
    saturacao: float = 2.0         # drive da saturação de fita (1 = quase limpa)
    wow: float = 0.0015            # profundidade da oscilação lenta (fração do tempo)
    reverb_voz: float = 0.18       # 0 = seco
    sidechain: float = 0.35        # quanto a base abaixa a cada bumbo
    volume_bateria: float = -21.0  # dB RMS
    volume_base: float = -21.0
    volume_voz: float = -18.0
    chiado: float = 0.006
    estalos: float = 3.0           # por segundo
    meio_tempo: bool = True        # permite bateria em meio-tempo em músicas rápidas
    encaixe_minimo: float = 0.25   # abaixo disso o loop não bate com a música: fica a bateria original


# ---------- efeitos ----------

def passa_baixa(y, fc, ordem=4):
    return signal.sosfiltfilt(signal.butter(ordem, min(fc, SR / 2 - 100), "lowpass", fs=SR, output="sos"), y)


def passa_alta(y, fc, ordem=2):
    return signal.sosfiltfilt(signal.butter(ordem, fc, "highpass", fs=SR, output="sos"), y)


def saturar(y, drive):
    pico = np.abs(y).max() + 1e-9
    return np.tanh(drive * y / pico) / np.tanh(drive) * pico


def wow_flutter(y, prof, rng):
    t = np.arange(len(y))
    fase = rng.uniform(0, 2 * np.pi)
    atraso = prof * SR * (np.sin(2 * np.pi * 0.55 * t / SR + fase) + 0.25 * np.sin(2 * np.pi * 6.0 * t / SR))
    return np.interp(t + atraso, t, y)


def reverb(y, mix, rt60=1.6, rng=None):
    if mix <= 0:
        return y
    rng = rng or np.random.default_rng(0)
    n = int(rt60 * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir = passa_baixa(ir, 6000)
    molhado = signal.fftconvolve(y, ir)[: len(y)]
    molhado *= np.sqrt(np.mean(y**2) / (np.mean(molhado**2) + 1e-12))
    return (1 - mix) * y + mix * molhado


def sidechain(y, bumbos, prof, soltar=0.18):
    """Abaixa y logo depois de cada bumbo e solta aos poucos."""
    env = np.ones(len(y))
    n = int(soltar * SR)
    curva = 1 - prof * np.exp(-np.arange(n) / (n / 4))
    for b in bumbos:
        i = int(b * SR)
        if i < len(y):
            m = min(n, len(y) - i)
            env[i : i + m] = np.minimum(env[i : i + m], curva[:m])
    return y * env


def vinil(n, chiado, estalos, rng):
    ruido = rng.standard_normal(n)
    ruido = signal.lfilter([0.049922, -0.095993, 0.050613, -0.004408], [1, -2.494956, 2.017265, -0.522189], ruido)
    ruido = passa_baixa(ruido, 7000)
    x = chiado * ruido / (np.abs(ruido).max() + 1e-9)
    for i in rng.integers(0, max(1, n - 200), int(estalos * n / SR)):
        x[i : i + 200] += rng.uniform(0.02, 0.1) * np.exp(-np.arange(200) / 25) * rng.choice([-1, 1])
    return x


def desacelerar(y, velocidade, manter_tom=True):
    """manter_tom=False: como um disco tocado mais devagar (mais lento e mais
    grave ao mesmo tempo). manter_tom=True: só mais lento, a voz não engrossa."""
    if abs(velocidade - 1) < 0.005:
        return y
    if manter_tom:
        return librosa.effects.time_stretch(y, rate=velocidade)
    return librosa.resample(y, orig_sr=SR, target_sr=int(round(SR / velocidade)))


def menos_graves(y, fc, db):
    """Prateleira de graves: tira `db` abaixo de fc sem cortar de vez."""
    graves = signal.sosfiltfilt(signal.butter(2, fc, "lowpass", fs=SR, output="sos"), y)
    return y + (10 ** (db / 20) - 1) * graves


def limitar(y, crista_db=9.0, ataque=0.002, soltar=0.08):
    """Compressor/limitador: segura os picos a no máximo `crista_db` acima do
    volume médio. Os loops de bateria têm picos ~20 dB acima da média; sem
    isso, uma caixa ou um aro mais forte vira um estalo por cima da música."""
    rms = np.sqrt(np.mean(y**2)) + 1e-12
    teto = rms * 10 ** (crista_db / 20)
    env = np.abs(y)
    a, s = np.exp(-1 / (ataque * SR)), np.exp(-1 / (soltar * SR))
    seg = signal.lfilter([1 - s], [1, -s], env)  # envelope (soltura)
    seg = np.maximum(env, seg)
    seg = signal.lfilter([1 - a], [1, -a], seg[::-1])[::-1]  # ataque antecipado, sem atraso
    ganho = np.minimum(1.0, teto / (seg + 1e-12))
    return np.tanh(y * ganho / teto) * teto  # o que ainda passar é arredondado


def nivel(y, db):
    return y * 10 ** (db / 20) / (np.sqrt(np.mean(y**2)) + 1e-12)


# ---------- bateria ----------

def desequilibrio(arquivo):
    """Quantas vezes o golpe mais forte do loop é maior que o golpe típico.
    Uns 1/3 dos loops do conjunto têm uma caixa ou aro 5-25x mais alto que o
    resto, que na mixagem vira um estalo."""
    y, _ = librosa.load(arquivo, sr=SR, mono=True)
    golpes = librosa.onset.onset_detect(y=y, sr=SR, units="samples", backtrack=True)
    picos = np.array([np.abs(y[g : g + 2000]).max() for g in golpes]) if len(golpes) else np.ones(1)
    return float(picos.max() / (np.median(picos) + 1e-9))


def listar_loops(pasta="dados/loops", max_desequilibrio=2.5):
    """(bpm, arquivo) dos loops equilibrados. A análise fica em cache em
    <pasta>/loops.json para não refazer a cada música."""
    import json
    cache_arq = os.path.join(pasta, "loops.json")
    cache = json.load(open(cache_arq)) if os.path.exists(cache_arq) else {}
    loops, mudou = [], False
    for f in sorted(glob.glob(os.path.join(pasta, "*.wav"))):
        nome = os.path.basename(f)
        m = re.search(r"(\d{2,3})\s*bpm", nome, re.I)
        if not m:
            continue
        if nome not in cache:
            cache[nome] = desequilibrio(f)
            mudou = True
        if cache[nome] <= max_desequilibrio:
            loops.append((float(m.group(1)), f))
    if mudou:
        json.dump(cache, open(cache_arq, "w"), indent=0)
    return loops


def grade_de_batidas(y, sr=SR):
    """Batidas da música (s) e o índice da batida que abre o compasso."""
    _, batidas = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=200)
    if len(batidas) < 8:
        raise ValueError("não achei batidas suficientes na música")
    # Cabeça do compasso: a fase (0-3) com mais energia de grave nas batidas.
    graves = librosa.onset.onset_strength(y=passa_baixa(y, 150), sr=sr)
    t_env = librosa.times_like(graves, sr=sr)
    forca = np.interp(batidas, t_env, graves)
    fase = int(np.argmax([forca[k::4].mean() for k in range(4)]))
    return batidas, fase


def montar_bateria(n, batidas, fase, loop_bpm, loop, meio_tempo):
    """Cola o loop compasso a compasso, esticando cada compasso para caber
    exatamente entre as batidas detectadas (assim acompanha variações de tempo).
    `loop` é o áudio do loop ou o caminho do arquivo."""
    if isinstance(loop, str):
        loop, _ = librosa.load(loop, sr=SR, mono=True)
    batida_loop = 60 / loop_bpm * SR  # amostras por batida no loop
    passo = 2 if meio_tempo else 1    # meio-tempo: uma batida do loop a cada 2 da música
    compasso = 4 * passo
    saida = np.zeros(n)
    bumbos = []
    k_loop = 0
    for i in range(fase, len(batidas) - compasso, compasso):
        ini, fim = int(batidas[i] * SR), int(batidas[i + compasso] * SR)
        a = int((k_loop % 4) * 4 * batida_loop)
        trecho = loop[a : a + int(4 * batida_loop)]
        if len(trecho) < 1000 or fim <= ini:
            continue
        esticado = librosa.resample(trecho, orig_sr=len(trecho), target_sr=fim - ini)[: fim - ini]
        # Rampas de 5 ms nas pontas: sem elas a emenda entre compassos estala.
        r = min(int(0.005 * SR), len(esticado) // 2)
        if r > 0:
            esticado[:r] *= np.linspace(0, 1, r)
            esticado[-r:] *= np.linspace(1, 0, r)
        m = min(len(esticado), n - ini)
        saida[ini : ini + m] += esticado[:m]
        bumbos += [batidas[i], batidas[i + 2 * passo]]
        k_loop += 1
    return saida, bumbos


HOP_ENC = 512


def _ataques(y):
    e = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_ENC)
    return e - e.mean()


def _correlacao(a, b, defasagem):
    """Correlação normalizada de a com b atrasado de `defasagem` quadros."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    x, y = (a[defasagem:], b[: n - defasagem]) if defasagem >= 0 else (a[: n + defasagem], b[-defasagem:])
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def encaixar(bateria_original, ritmo, loops, permitir_meio_tempo=True, indice_loop=0):
    """Escolhe andamento, início do compasso e um ajuste fino para a bateria
    nova bater junto com a bateria original da música.

    O detector de batidas sozinho erra com frequência: pega 2/3 ou 3/2 do
    andamento em músicas com swing, ou cai uma batida fora do compasso. Aqui
    testo vários andamentos candidatos, todas as posições do início do
    compasso e defasagens de até 1/4 de batida, monto a bateria para cada
    combinação e meço a correlação dos ataques com os da bateria original.
    Fica a combinação com maior correlação.
    """
    ref = _ataques(bateria_original)
    t0, _ = librosa.beat.beat_track(y=ritmo, sr=SR, tightness=200)
    t0 = float(np.atleast_1d(t0)[0])
    base = [t0 * f for f in (1, 2 / 3, 3 / 4, 4 / 3, 3 / 2, 0.5, 2)]
    # Também os períodos que mais se repetem nos ataques da bateria original
    # (o detector às vezes nem chega perto do andamento certo).
    ac = librosa.autocorrelate(_ataques(bateria_original) + 1e-9, max_size=int(3 * SR / HOP_ENC))
    periodos = np.arange(len(ac)) * HOP_ENC / SR
    validos = np.flatnonzero((periodos > 0.28) & (periodos < 2.6))
    for i in validos[np.argsort(ac[validos])[-4:]]:
        base += [60 / periodos[i] * f for f in (1, 2, 0.5)]
    candidatos = sorted({round(c, 1) for c in base if 50 <= c <= 200})
    audio_loops = {}
    melhor = dict(nota=-np.inf)
    vistos = set()
    for c in candidatos:
        _, batidas = librosa.beat.beat_track(y=ritmo, sr=SR, start_bpm=c, tightness=400, units="time")
        if len(batidas) < 9:
            continue
        tempo = 60 / np.median(np.diff(batidas))
        if round(tempo) in vistos:
            continue
        vistos.add(round(tempo))
        for meio in ([False, True] if permitir_meio_tempo and tempo > 95 else [False]):
            alvo = tempo / 2 if meio else tempo
            if not 45 <= alvo <= 95:
                continue
            proximos = sorted(loops, key=lambda l: abs(np.log(l[0] / alvo)))
            bpm, arq = proximos[min(indice_loop, len(proximos) - 1)]
            if arq not in audio_loops:
                audio_loops[arq] = librosa.load(arq, sr=SR, mono=True)[0]
            batida_q = 60 / tempo * SR / HOP_ENC
            for fase in range(4 * (2 if meio else 1)):
                tr, _ = montar_bateria(len(bateria_original), batidas, fase, bpm, audio_loops[arq], meio)
                novo = _ataques(tr)
                for d in range(-int(batida_q / 4), int(batida_q / 4) + 1):
                    nota = _correlacao(ref, novo, d)
                    if nota > melhor["nota"]:
                        melhor = dict(nota=nota, batidas=batidas + d * HOP_ENC / SR, fase=fase,
                                      meio=meio, tempo=tempo, alvo=alvo, loop_bpm=bpm, loop=arq)
    if not np.isfinite(melhor["nota"]):
        raise ValueError("não consegui encaixar a bateria nesta música")
    return melhor


# ---------- montagem ----------

def produzir(partes, ajustes=None, semente=0, pasta_loops="dados/loops"):
    """partes: dict com 'vocals', 'drums', 'bass', 'other' (mono, 44,1 kHz)."""
    a = ajustes or Ajustes()
    rng = np.random.default_rng(semente)
    voz, baixo, resto = (desacelerar(partes[k], a.velocidade, a.manter_tom) for k in ("vocals", "bass", "other"))
    n = min(len(voz), len(baixo), len(resto))
    voz, baixo, resto = voz[:n], baixo[:n], resto[:n]
    harmonia = baixo + resto

    # O encaixe é feito na música original (desacelerar antes borra os ataques
    # e o detector erra); depois é só converter os tempos para a versão lenta.
    loops = listar_loops(pasta_loops)
    if not loops:
        raise FileNotFoundError(f"nenhum loop em {pasta_loops}")
    enc = encaixar(partes["drums"], partes["drums"] + partes["bass"] + partes["other"], loops, a.meio_tempo,
                   indice_loop=int(rng.integers(0, 3)))
    batidas, fase, meio = enc["batidas"] / a.velocidade, enc["fase"], enc["meio"]
    tempo = enc["tempo"] * a.velocidade
    loop_bpm, loop_arq = enc["loop_bpm"], enc["loop"]
    bateria, bumbos = montar_bateria(n, batidas, fase, loop_bpm, loop_arq, meio)
    if enc["nota"] < a.encaixe_minimo:
        # Nenhum loop bate com a música (swing, quebra de ritmo...): melhor
        # manter a bateria original, só com o tratamento lo-fi.
        bateria = desacelerar(partes["drums"], a.velocidade, a.manter_tom)[:n]
        bateria = saturar(passa_baixa(np.pad(bateria, (0, n - len(bateria))), 5000), 1.8)
        loop_arq, loop_bpm = "bateria original (nenhum loop encaixou)", 0
        bumbos = []  # sem grade confiável, sem sidechain (senão a base "respira" fora do tempo)

    base = passa_alta(harmonia, 35)
    base = passa_baixa(base, a.corte_base)
    base = saturar(base, a.saturacao)
    base = wow_flutter(base, a.wow, rng)
    base = sidechain(base, bumbos, a.sidechain)

    voz = passa_alta(voz, 90)
    voz = reverb(voz, a.reverb_voz, rng=rng)

    bateria = limitar(passa_baixa(bateria, a.corte_bateria))
    mix = nivel(voz, a.volume_voz) + nivel(base, a.volume_base) + nivel(bateria, a.volume_bateria)
    mix = menos_graves(mix, a.corte_graves, a.graves_db)
    mix = passa_baixa(mix, a.corte_final, ordem=2)
    mix = mix + vinil(n, a.chiado, a.estalos, rng)
    info = dict(tempo=round(float(tempo), 1), meio_tempo=bool(meio), encaixe=round(enc["nota"], 2),
                loop=os.path.basename(loop_arq), loop_bpm=loop_bpm)
    return (0.9 * mix / (np.abs(mix).max() + 1e-9)).astype(np.float32), info
