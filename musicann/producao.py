"""Monta a versão lo-fi de uma música do jeito que um produtor faria.

A música é separada em voz, bateria, baixo e o resto (Demucs). Depois:
  - voz: fica limpa, com um reverb leve
  - bateria: a original sai e entra um loop de bateria lo-fi de verdade,
    encaixado compasso a compasso nas batidas da música
  - baixo e o resto: corte de agudos, saturação de fita, oscilação lenta
    (wow/flutter) e o volume "respirando" com o bumbo (sidechain)
  - tudo desacelerado como um disco mais lento, com chiado de vinil por cima

Os ajustes ficam em `Ajustes`; as seções e automações, em musicann/arranjo.py.
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
    # 1 = sem desacelerar; 0.95 = 5% mais lento (~0,9 semitom abaixo). As lo-fi
    # do YouTube ficam em 0,92 (mediana, dados/estudo/); 0,88 já engrossa demais a voz.
    velocidade: float = 0.95
    # False: como disco tocado mais devagar (mais grave junto), sem nenhum
    # artefato. True: mantém o tom, mas o esticamento (phase vocoder) borra os
    # ataques e deixa o som "lavado"; por isso o padrão é False com só 3%.
    manter_tom: bool = False
    corte_base: float = 4500.0     # Hz, passa-baixa no baixo e no resto (o "abafado")
    corte_final: float = 11000.0   # Hz, passa-baixa suave na mixagem toda
    corte_bateria: float = 7000.0  # Hz, bateria um pouco abafada para casar com a base
    corte_graves: float = 120.0    # Hz, abaixo disso a mixagem perde um pouco (menos "grossa")
    graves_db: float = -2.0
    saturacao: float = 2.0         # drive da saturação de fita (1 = quase limpa)
    wow: float = 0.0015            # profundidade da oscilação lenta (fração do tempo)
    reverb_voz: float = 0.18       # 0 = seco
    voz_sala: float = 0.14         # mínimo de reverb na voz em qualquer seção
    reverb_jogado: float = 0.55    # reverb na última palavra antes de cada virada
    voz_agudos_hz: float = 6500.0  # acima disso a voz fica um pouco mais macia
    voz_agudos_db: float = -4.0
    voz_saturacao: float = 1.3     # saturação de fita leve na voz
    sidechain: float = 0.35        # quanto a base abaixa a cada bumbo
    # Volumes (dB RMS). Nas lo-fi do YouTube a base fica ~5 dB mais forte em
    # relação à voz do que na original.
    volume_bateria: float = -22.0
    bateria_original_db: float = -13.0  # bateria original tratada por baixo do loop (cola o groove)
    volume_base: float = -18.0
    volume_voz: float = -19.0
    # A bateria e a base seguem o equilíbrio da música original em relação à
    # voz, `mais_que_original` dB mais fortes (as lo-fi do YouTube: +5), sem
    # ficar abaixo dos volumes acima nem passar dos tetos (a voz continua na
    # frente). Neighbors tem bateria 4 dB acima da voz; Peace, 10 dB abaixo.
    mais_que_original: float = 2.0
    teto_bateria: float = 0.0      # dB acima da voz, no máximo
    teto_base: float = 2.0
    chiado: float = 0.010          # as lo-fi do YouTube têm ~6 dB a mais de chiado
    estalos: float = 3.0           # por segundo
    meio_tempo: bool = True        # permite bateria em meio-tempo em músicas rápidas
    encaixe_minimo: float = 0.25   # abaixo disso o loop não bate com a música: fica a bateria original
    pratos_db: float = -3.0        # pratos e chimbal da bateria original, em relação ao volume que tinham na música
    pratos_hz: float = 4500.0      # acima disso, a bateria original fica (pratos, chimbal, o "tss")
    # "auto": loop em música de andamento fixo, substituição nas outras;
    # "loop" ou "substituicao" forçam um dos dois.
    bateria: str = "auto"
    # Teclado novo (musicann/harmonia.py): os acordes da música num piano
    # elétrico, no lugar do resto original (teclados, samples, guitarras),
    # como fazem os remixes lo-fi do YouTube.
    teclado: bool = True
    teclado_db: float = 0.0        # volume do teclado em relação ao resto original
    resto_original_db: float = -18.0  # quanto o resto original abaixa onde o teclado toca
    baixo_novo: bool = True        # baixo novo colado no bumbo do loop
    baixo_novo_db: float = 0.0     # volume do baixo novo em relação ao baixo original
    baixo_original_db: float = -15.0  # quanto o baixo original abaixa onde o baixo novo toca
    # O gancho do teclado/sample original tocado no piano elétrico (--riff).
    riff: bool = False
    riff_db: float = -4.0
    piano_sob_voz_db: float = -5.0 # com o riff: o piano abaixa isso enquanto o rapper canta
    # Quando o resto original é um ciclo que se repete, o --riff toca esse
    # ciclo como uma frase (harmonia.gancho). A frase substitui o sample, então
    # fica perto do volume dele e quase não recua sob a voz; mais baixa que
    # isso, some na mixagem e a música volta a soar como a original.
    gancho_db: float = -1.5
    gancho_sob_voz_db: float = -0.5
    # Varia o arranjo da frase por seção e a cada 8 compassos (harmonia.gancho).
    gancho_variar: bool = True
    # Volume mínimo da frase contra a voz onde o rapper canta (antes da
    # mixagem; ver o piso em produzir). None desliga.
    gancho_piso_voz_db: float = -5.0
    # Com o riff: "gancho" transcreve o sample que se repete; "progressao"
    # toca uma progressão simples nos acordes da música (harmonia.progressao).
    # Os dois usam o mesmo volume (gancho_db, gancho_sob_voz_db e o piso).
    teclado_estilo: str = "gancho"
    # Chorus, sala e agudos amaciados no piano elétrico (cor_do_teclado).
    cor_teclado: bool = True
    # Som do piano elétrico: "sintetizado" (harmonia.rhodes) ou "gravado"
    # (amostras de um Rhodes de verdade, harmonia.rhodes_gravado; precisa de
    # dados/instrumentos, fora do git).
    timbre: str = "sintetizado"
    perda_maxima_db: float = 3.0   # a base nunca fica mais que isso abaixo do equilíbrio original


# Perfis por estilo. "rap" é a base refeita (a lo-fi de Peace, tag
# v3-peace). "brasileira" volta ao jeito anterior, com a base original
# filtrada: nas brasileiras (samba, MPB, soul tocado por banda) a graça está
# no violão, no suingue e na banda, e trocar isso por piano elétrico e
# batida de hip-hop descaracteriza a música. Ficam as correções que valem
# para todas (buracos da voz, pratos, bateria original abafada onde o loop
# não toca).
PERFIS = {
    "rap": {},
    "brasileira": dict(teclado=False, baixo_novo=False, velocidade=0.97, chiado=0.006, voz_sala=0.10,
                       volume_bateria=-25.0, volume_base=-21.0, volume_voz=-18.0, mais_que_original=-99.0,
                       bateria_original_db=-9.0),
    # Rap mais pesado (808 na frente, trap), como Chill Bill. A lo-fi mais
    # ouvida dela no YouTube (LLusion, 2,4 mi) não refaz a base: é a própria
    # gravação a 0,78 da velocidade (o tom desce junto), com a mistura duas
    # vezes mais escura (corte em 1,3 kHz contra 2,6 kHz) e a voz 7 dB mais
    # atrás e mais escura; bateria, 808 e assobio são os originais. Aqui:
    # o mesmo caminho, um pouco menos lento e com a voz ainda clara.
    "pesada": dict(teclado=False, baixo_novo=False, riff=False, bateria="original", velocidade=0.80,
                   corte_base=2500.0, voz_agudos_hz=4500.0, voz_agudos_db=-6.0,
                   voz_sala=0.22, chiado=0.016, mais_que_original=3.0, teto_base=20.0, teto_bateria=20.0,
                   corte_final=8000.0, pratos_db=2.0,  # o chimbal acompanha a bateria, que fica 5 dB mais forte
                   graves_db=-4.5),  # 808 um pouco menos marcante
}


def ajustes_do_perfil(nome="rap", **extra):
    a = Ajustes(**PERFIS[nome])
    for k, v in extra.items():
        setattr(a, k, v)
    return a


# ---------- efeitos ----------

def passa_baixa(y, fc, ordem=4):
    return signal.sosfiltfilt(signal.butter(ordem, min(fc, SR / 2 - 100), "lowpass", fs=SR, output="sos"), y)


def passa_alta(y, fc, ordem=2):
    return signal.sosfiltfilt(signal.butter(ordem, fc, "highpass", fs=SR, output="sos"), y)


def saturar(y, drive):
    pico = np.abs(y).max() + 1e-9
    return np.tanh(drive * y / pico) / np.tanh(drive) * pico


def mono(y):
    return y.mean(0) if y.ndim == 2 else y


def wow_flutter(y, prof, rng):
    """Oscilação lenta de velocidade, como fita. Funciona em mono ou estéreo
    (os dois canais oscilam juntos, como numa fita de verdade)."""
    t = np.arange(y.shape[-1])
    fase = rng.uniform(0, 2 * np.pi)
    atraso = prof * SR * (np.sin(2 * np.pi * 0.55 * t / SR + fase) + 0.25 * np.sin(2 * np.pi * 6.0 * t / SR))
    if y.ndim == 2:
        return np.stack([np.interp(t + atraso, t, c) for c in y])
    return np.interp(t + atraso, t, y)


def cor_do_teclado(y, rng=None):
    """O tratamento do piano elétrico nas lo-fis: chorus leve (cada canal com
    um atraso de ~7 ms oscilando devagar, o "vai e vem" do Rhodes com efeito),
    um pouco de sala e os agudos amaciados. Mesmo volume (RMS) da entrada,
    para não mexer nos níveis já ajustados (gancho_db, piso)."""
    rng = rng or np.random.default_rng(0)
    if not np.any(y):
        return y
    t = np.arange(y.shape[1]) / SR
    saida = np.empty_like(y)
    for c in range(2):
        atraso = (0.007 + 0.0015 * np.sin(2 * np.pi * (0.55 + 0.1 * c) * t + rng.uniform(0, 2 * np.pi))) * SR
        saida[c] = 0.8 * y[c] + 0.35 * np.interp(np.arange(y.shape[1]) - atraso, np.arange(y.shape[1]), y[c])
    saida = reverb(saida, 0.15, rt60=0.9, rng=rng)
    saida = passa_baixa(saida, 5500, ordem=2)
    return saida * 10 ** ((rms_db(y) - rms_db(saida)) / 20)


def reverb(y, mix, rt60=1.6, rng=None):
    """Reverb por convolução com ruído decaindo. Em estéreo, cada canal tem
    uma resposta diferente, o que abre o som (o reverb fica "largo")."""
    if mix <= 0:
        return y
    rng = rng or np.random.default_rng(0)
    n = int(rt60 * SR)
    canais = y if y.ndim == 2 else y[None]
    saida = []
    for c in canais:
        ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        ir = passa_baixa(ir, 6000)
        molhado = signal.fftconvolve(c, ir)[: len(c)]
        molhado *= np.sqrt(np.mean(c**2) / (np.mean(molhado**2) + 1e-12))
        saida.append((1 - mix) * c + mix * molhado)
    return np.stack(saida) if y.ndim == 2 else saida[0]


def sidechain(y, bumbos, prof, soltar=0.18):
    """Abaixa y logo depois de cada bumbo e solta aos poucos."""
    n_y = y.shape[-1]
    env = np.ones(n_y)
    n = int(soltar * SR)
    curva = 1 - prof * np.exp(-np.arange(n) / (n / 4))
    for b in bumbos:
        i = int(b * SR)
        if i < n_y:
            m = min(n, n_y - i)
            env[i : i + m] = np.minimum(env[i : i + m], curva[:m])
    return y * env


def vinil(n, chiado, estalos, rng):
    """Chiado (independente em cada canal) e estalos (no centro), estéreo."""
    ruido = rng.standard_normal((2, n))
    ruido = signal.lfilter([0.049922, -0.095993, 0.050613, -0.004408], [1, -2.494956, 2.017265, -0.522189], ruido)
    ruido = passa_baixa(ruido, 7000)
    x = chiado * ruido / (np.abs(ruido).max() + 1e-9)
    for i in rng.integers(0, max(1, n - 200), int(estalos * n / SR)):
        x[:, i : i + 200] += rng.uniform(0.02, 0.1) * np.exp(-np.arange(200) / 25) * rng.choice([-1, 1])
    return x


def desacelerar(y, velocidade, manter_tom=True):
    """manter_tom=False: como um disco tocado mais devagar (mais lento e mais
    grave ao mesmo tempo). manter_tom=True: só mais lento, a voz não engrossa."""
    if abs(velocidade - 1) < 0.005:
        return y
    if manter_tom:
        return librosa.effects.time_stretch(y, rate=velocidade)
    return librosa.resample(y, orig_sr=SR, target_sr=int(round(SR / velocidade)))


def menos_agudos(y, fc, db):
    """Prateleira de agudos: tira `db` acima de fc sem cortar de vez (deixa
    macio, sem o "abafado" de um passa-baixa)."""
    agudos = signal.sosfiltfilt(signal.butter(2, fc, "highpass", fs=SR, output="sos"), y)
    return y + (10 ** (db / 20) - 1) * agudos


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
    seg = signal.lfilter([1 - a], [1, -a], seg[..., ::-1])[..., ::-1]  # ataque antecipado, sem atraso
    ganho = np.minimum(1.0, teto / (seg + 1e-12))
    return np.tanh(y * ganho / teto) * teto  # o que ainda passar é arredondado


def nivel(y, db):
    return y * 10 ** (db / 20) / (np.sqrt(np.mean(y**2)) + 1e-12)


def rms_db(y):
    return 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-12)


def camada_pratos(bateria_original, corte=4500.0):
    """Pratos e chimbal da bateria original (o "tss" que marca a música),
    com um toque de fita. O loop lo-fi entra no lugar do bumbo e da caixa,
    mas os pratos são parte da identidade da música; e é nessa faixa que o
    Demucs joga os "s" e "t" da voz que vazam para a bateria, que assim
    voltam também."""
    topo = passa_alta(bateria_original, corte, ordem=4)
    topo = passa_baixa(topo, 9500, ordem=2)
    return saturar(topo, 1.5)


def rede_de_seguranca(bateria, bateria_original, duracao, n, janela=8.0):
    """A cada `janela` s, se o bumbo e a caixa da bateria nova não batem com
    os da bateria original (atraso típico > 35 ms), a bateria nova sai
    daquele trecho. Melhor um trecho com a bateria original abafada que um
    trecho descompassado. Devolve (curva 0-1, trechos tirados)."""
    seguro = np.ones(n)
    ref = golpes_fortes(bateria_original, 0.5)
    fortes = golpes_fortes(bateria, 0.4)
    mudos = 0
    if len(ref) <= 8 or len(fortes) <= 8:
        return seguro, mudos
    t, d = _atrasos(fortes, ref, janela=0.12)
    for ini in np.arange(0, duracao, janela):
        dentro = (t >= ini) & (t < ini + janela)
        golpes = ((fortes >= ini) & (fortes < ini + janela)).sum()
        originais = ((ref >= ini) & (ref < ini + janela)).sum()
        # Só julga onde a bateria original toca de fato (>= 4 golpes fortes):
        # trecho sem bateria original não é prova de descompasso.
        if golpes >= 4 and originais >= 4 and (dentro.sum() < golpes / 3 or np.median(np.abs(d[dentro])) * 1000 > 35):
            seguro[int(ini * SR) : int(min(duracao, ini + janela) * SR)] = 0
            mudos += 1
    return seguro, mudos


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


def montar_bateria(n, batidas, fase, loop_bpm, loop, meio_tempo, swing_musica=None, swing_loop=0.5):
    """Cola o loop batida a batida, esticando cada batida do loop para caber
    exatamente entre duas batidas da música (acompanha as variações de tempo
    de gravações tocadas ao vivo). `loop` é o áudio do loop ou o caminho.

    swing_musica / swing_loop: onde cai o contratempo dentro da batida (0,5 =
    reto, ~0,67 = tercina). Se a música tem swing e o loop não, cada batida do
    loop é entortada em duas partes para o contratempo cair onde cai na
    música (senão, metade dos golpes fica ~50 ms fora da guitarra)."""
    if isinstance(loop, str):
        loop, _ = librosa.load(loop, sr=SR, mono=False)
    if loop.ndim == 1:
        loop = np.stack([loop, loop])
    batida_loop = 60 / loop_bpm * SR  # amostras por batida no loop
    passo = 2 if meio_tempo else 1    # meio-tempo: uma batida do loop a cada 2 da música
    entortar = swing_musica is not None and not meio_tempo and abs(swing_musica - swing_loop) > 0.02
    saida = np.zeros((2, n))
    bumbos = []
    k = 0  # batida dentro do loop (o loop tem 16 batidas: 4 compassos)
    for i in range(fase, len(batidas) - passo, passo):
        t0, t1 = batidas[i], batidas[i + passo]
        j = k % 16
        a = j * batida_loop
        if entortar:
            partes = [(a, a + swing_loop * batida_loop, t0, t0 + swing_musica * (t1 - t0)),
                      (a + swing_loop * batida_loop, a + batida_loop, t0 + swing_musica * (t1 - t0), t1)]
        else:
            partes = [(a, a + batida_loop, t0, t1)]
        for a0, a1, s0, s1 in partes:
            p0, p1 = int(round(s0 * SR)), int(round(s1 * SR))
            trecho = loop[:, int(round(a0)) : int(round(a1))]
            if trecho.shape[1] < 64 or p1 <= p0 or p0 >= n:
                continue
            esticado = librosa.resample(trecho, orig_sr=trecho.shape[1], target_sr=p1 - p0)[:, : p1 - p0]
            m = min(esticado.shape[1], n - p0)
            saida[:, p0 : p0 + m] += esticado[:, :m]
        if j % 4 in (0, 2):
            bumbos.append(t0)
        k += 1
    return saida, bumbos


def medir_swing(ataques, batidas):
    """Onde cai o contratempo dentro da batida (0,5 = reto, 0,67 = tercina),
    pela mediana dos ataques entre 35% e 80% da batida."""
    if len(batidas) < 4 or len(ataques) < 10:
        return 0.5
    idx = np.searchsorted(batidas, ataques) - 1
    ok = (idx >= 0) & (idx < len(batidas) - 1)
    frac = (ataques[ok] - batidas[idx[ok]]) / (batidas[idx[ok] + 1] - batidas[idx[ok]])
    meio = frac[(frac > 0.35) & (frac < 0.8)]
    return float(np.clip(np.median(meio), 0.5, 0.7)) if len(meio) >= 10 else 0.5


def swing_do_loop(loop, loop_bpm):
    if isinstance(loop, str):
        loop, _ = librosa.load(loop, sr=SR, mono=True)
    ataques = librosa.onset.onset_detect(y=mono(loop), sr=SR, units="time", backtrack=True)
    grade = np.arange(0, len(mono(loop)) / SR + 60 / loop_bpm, 60 / loop_bpm)
    return medir_swing(ataques, grade)


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


def grade_regular(batidas, tolerancia=0.025, minimo=0.85):
    """Se as batidas detectadas cabem numa grade de andamento fixo (desvio
    menor que `tolerancia` segundos), devolve a grade perfeitamente regular.

    Músicas produzidas no computador têm andamento fixo, mas o detector
    oscila 20-40 ms de uma batida para outra. Como o loop é esticado
    compasso a compasso seguindo as batidas, essa oscilação deixava a
    bateria "quase descompassada". Músicas tocadas ao vivo, com andamento
    que varia de verdade, mantêm a grade detectada."""
    i = np.arange(len(batidas))
    p = np.polyfit(i, batidas, 1)
    regular = np.polyval(p, i)
    if np.std(batidas - regular) < tolerancia:
        return regular
    # Um tropeço do detector (um trecho em que ele perde a batida) não deve
    # mandar a música toda para a substituição: em produções de estúdio ~90%
    # das batidas caem a menos de 25 ms de uma grade fixa, contra 10-15% nas
    # gravações ao vivo. A reta é refeita sem os tropeços; onde o loop ainda
    # não bater, a rede de segurança tira o loop do trecho.
    reta = _reta_sem_tropecos(batidas, tolerancia)
    if reta is None:
        return batidas
    return reta if np.mean(np.abs(batidas - reta) < tolerancia) >= minimo else batidas


def _reta_sem_tropecos(batidas, tolerancia=0.025):
    """A grade fixa que mais batidas seguem: a reta refeita sem as que fogem
    mais de 3x a tolerância (ou None, se sobrarem poucas)."""
    i = np.arange(len(batidas))
    usar = np.ones(len(batidas), bool)
    for _ in range(3):
        if usar.sum() < 9:
            return None
        p = np.polyfit(i[usar], batidas[usar], 1)
        usar = np.abs(batidas - np.polyval(p, i)) < 3 * tolerancia
    return np.polyval(p, i)


def na_grade(batidas, tolerancia=0.025):
    """Fração das batidas a menos de `tolerancia` s da grade fixa."""
    reta = _reta_sem_tropecos(np.asarray(batidas), tolerancia)
    return 0.0 if reta is None else float(np.mean(np.abs(batidas - reta) < tolerancia))


def _bandas(y):
    """Curvas de ataque do bumbo (< 150 Hz) e da caixa/chimbal (200 Hz-6 kHz)."""
    grave = signal.sosfilt(signal.butter(4, 150, "lowpass", fs=SR, output="sos"), y)
    medio = signal.sosfilt(signal.butter(2, [200, 6000], "bandpass", fs=SR, output="sos"), y)
    curvas = [librosa.onset.onset_strength(y=s, sr=SR, hop_length=HOP_ENC) for s in (grave, medio)]
    return np.stack([c - c.mean() for c in curvas])


def _bandas_montadas(env_loop, loop_bpm, batidas, fase, meio_tempo, n_quadros):
    """Curvas de ataque que o loop teria se fosse montado na grade de batidas
    (mesma colagem de montar_bateria, mas sem gerar áudio: bem mais rápido)."""
    q_batida = 60 / loop_bpm * SR / HOP_ENC
    passo = 2 if meio_tempo else 1
    compasso = 4 * passo
    saida = np.zeros((2, n_quadros))
    k = 0
    for i in range(fase, len(batidas) - compasso, compasso):
        f0, f1 = batidas[i] * SR / HOP_ENC, batidas[i + compasso] * SR / HOP_ENC
        quadros = np.arange(int(np.ceil(f0)), min(int(np.ceil(f1)), n_quadros))
        if len(quadros) == 0:
            continue
        a0 = (k % 4) * 4 * q_batida
        origem = a0 + (quadros - f0) / (f1 - f0) * 4 * q_batida
        for b in range(2):
            saida[b, quadros] = np.interp(origem, np.arange(env_loop.shape[1]), env_loop[b], right=0)
        k += 1
    return saida


def encaixar(bateria_original, ritmo, loops, permitir_meio_tempo=True, n_loops=20):
    """Escolhe andamento, loop, início do compasso e ajuste fino para a
    bateria nova combinar com a bateria original da música.

    O detector de batidas sozinho erra com frequência (pega 2/3 ou 3/2 do
    andamento, ou cai uma batida fora do compasso), e o loop mais próximo no
    andamento nem sempre tem um padrão que combine (um boom-bap sobre o bumbo
    em todas as batidas do synth-pop briga mesmo começando no lugar certo).
    Então testo:
      - vários andamentos candidatos (frações do detectado e os períodos que
        mais se repetem na bateria original), com e sem meio-tempo;
      - os `n_loops` loops mais próximos do andamento;
      - todas as posições do início do compasso;
      - defasagens de até 1/4 de batida;
    e meço quanto o bumbo do loop coincide com o bumbo original e a caixa com
    a caixa. Devolve a melhor combinação e o segundo melhor loop na mesma
    grade (para variar no refrão).
    """
    ref = _bandas(bateria_original)
    n_q = ref.shape[1]
    t0, _ = librosa.beat.beat_track(y=ritmo, sr=SR, tightness=200)
    t0 = float(np.atleast_1d(t0)[0])
    base = [t0 * f for f in (1, 2 / 3, 3 / 4, 4 / 3, 3 / 2, 0.5, 2)]
    ac = librosa.autocorrelate(_ataques(bateria_original) + 1e-9, max_size=int(3 * SR / HOP_ENC))
    periodos = np.arange(len(ac)) * HOP_ENC / SR
    validos = np.flatnonzero((periodos > 0.28) & (periodos < 2.6))
    for i in validos[np.argsort(ac[validos])[-4:]]:
        base += [60 / periodos[i] * f for f in (1, 2, 0.5)]
    candidatos = sorted({round(c, 1) for c in base if 50 <= c <= 200})

    env_loops, densidade = {}, {}
    resultados = []  # (nota, chave da grade, arq, bpm, fase, defasagem)
    grades, vistos = {}, set()
    for c in candidatos:
        _, batidas = librosa.beat.beat_track(y=ritmo, sr=SR, start_bpm=c, tightness=400, units="time")
        if len(batidas) < 9:
            continue
        batidas = grade_regular(batidas)
        tempo = 60 / np.median(np.diff(batidas))
        if round(tempo) in vistos:
            continue
        vistos.add(round(tempo))
        for meio in ([False, True] if permitir_meio_tempo and tempo > 95 else [False]):
            alvo = tempo / 2 if meio else tempo
            if not 45 <= alvo <= 95:
                continue
            chave = (round(tempo), meio)
            grades[chave] = dict(batidas=batidas, tempo=tempo, meio=meio, alvo=alvo)
            q_batida = 60 / tempo * SR / HOP_ENC
            tol = int(q_batida / 4)
            for bpm, arq in sorted(loops, key=lambda l: abs(np.log(l[0] / alvo)))[:n_loops]:
                if arq not in env_loops:
                    y_loop = librosa.load(arq, sr=SR, mono=True)[0]
                    env_loops[arq] = _bandas(y_loop)
                    densidade[arq] = len(librosa.onset.onset_detect(y=y_loop, sr=SR))
                for fase in range(4 * (2 if meio else 1)):
                    novo = _bandas_montadas(env_loops[arq], bpm, batidas, fase, meio, n_q)
                    melhor_d, melhor_n = 0, -np.inf
                    for d in range(-tol, tol + 1):
                        nota = 0.5 * (_correlacao(ref[0], novo[0], d) + _correlacao(ref[1], novo[1], d))
                        if nota > melhor_n:
                            melhor_d, melhor_n = d, nota
                    resultados.append((melhor_n, chave, arq, bpm, fase, melhor_d))
    if not resultados:
        raise ValueError("não consegui encaixar a bateria nesta música")
    melhor_nota = max(r[0] for r in resultados)
    # Entre os que encaixam quase tão bem quanto o melhor (>= 93%), prefere o
    # loop com mais golpes (chimbal, ghost notes): soa menos "batida simples".
    quase = [r for r in resultados if r[0] >= 0.93 * melhor_nota]
    nota, chave, arq, bpm, fase, d = max(quase, key=lambda r: (densidade[r[2]], r[0]))
    g = grades[chave]
    # Segundo loop: o melhor outro loop na mesma grade e no mesmo início de compasso.
    outros = [r for r in resultados if r[1] == chave and r[4] == fase and r[2] != arq]
    segundo = max(outros, key=lambda r: r[0]) if outros else (nota, chave, arq, bpm, fase, d)
    return dict(nota=nota, batidas=g["batidas"] + d * HOP_ENC / SR, fase=fase, meio=g["meio"],
                tempo=g["tempo"], alvo=g["alvo"], loop=arq, loop_bpm=bpm,
                loop2=segundo[2], loop2_bpm=segundo[3], nota2=segundo[0])


# ---------- montagem ----------

def produzir(partes, ajustes=None, semente=0, pasta_loops="dados/loops", variar=True, secoes_externas=None, camadas=None):
    """partes: dict com 'vocals', 'drums', 'bass', 'other' (mono ou estéreo, 44,1 kHz).

    Com variar=True, muda o tratamento ao longo da música, como um produtor
    faria com automação (valores em arranjo.PERFIL):
      introdução    sem bateria, filtro da base abrindo aos poucos
      verso         loop principal abafado, base fechada, voz seca
      refrão        segundo loop aberto, base aberta, voz com reverb
      ponte         sem bateria, base fechada, voz bem molhada
      instrumental  segundo loop, base aberta
      viradas       meio compasso sem bateria e filtro fechando antes do
                    refrão; reverb jogado na última palavra
      final         bateria sai em 2 compassos, filtro fechando até ~450 Hz
    secoes_externas: lista {ini, fim, papel} na linha do tempo original (por
    exemplo, do SongFormer); sem ela usa a detecção simples de arranjo.py.
    camadas: dict opcional que recebe cada camada da mixagem (no tempo da
    lo-fi) e a grade de batidas, para medir o encaixe entre elas.
    """
    from . import arranjo, reparo
    a = ajustes or Ajustes()
    rng = np.random.default_rng(semente)
    # Áudio em estéreo (2, n); análises (batidas, seções) em mono.
    estereo = {k: (v if v.ndim == 2 else np.stack([v, v])) for k, v in partes.items()}
    pm = {k: mono(v) for k, v in partes.items()}
    # Em cada golpe forte da bateria original, a separação deixa um buraco na
    # voz e no resto, que a bateria nova não tapa (musicann/reparo.py).
    golpes_orig = golpes_fortes(pm["drums"], 0.5)
    for k in ("vocals", "other"):
        estereo[k] = reparo.tapar_buracos(estereo[k], golpes_orig, SR)
    # Assobio (ou flauta) que a separação pôs na voz: vai para a base, onde
    # recebe o tratamento lo-fi em vez de passar limpo.
    estereo["vocals"], assobio = reparo.separar_assobio(estereo["vocals"], SR)
    tem_assobio = bool(np.any(assobio))
    voz, baixo, resto, assobio = (desacelerar(x, a.velocidade, a.manter_tom)
                                  for x in (estereo["vocals"], estereo["bass"], estereo["other"], assobio))
    n = min(voz.shape[1], baixo.shape[1], resto.shape[1])
    voz, baixo, resto = voz[:, :n], baixo[:, :n], resto[:, :n]
    assobio = np.pad(assobio[:, :n], ((0, 0), (0, max(0, n - assobio.shape[1]))))
    harmonia = baixo + resto
    duracao = n / SR

    # Encaixe na música original (desacelerar antes borra os ataques); depois
    # é só converter os tempos para a versão lenta.
    loops = listar_loops(pasta_loops)
    if not loops:
        raise FileNotFoundError(f"nenhum loop em {pasta_loops}")
    ritmo = pm["drums"] + pm["bass"] + pm["other"]
    enc = encaixar(pm["drums"], ritmo, loops, a.meio_tempo)
    # Ajuste fino golpe a golpe: o bumbo e a caixa do loop são alinhados com o
    # bumbo e a caixa da bateria original. O loop só é usado em música de
    # andamento fixo (gravação moderna, bateria bem separada pelo Demucs);
    # gravações antigas ou tocadas ao vivo vão para a substituição de bateria.
    # (Alinhar pelo baixo e violão errava no rap: teclados e samples quase não
    # têm ataques marcados.)
    ref_alin = pm["drums"]
    f0, bpm0, lp0, meio0 = enc["fase"], enc["loop_bpm"], enc["loop"], enc["meio"]
    bat_ref = afinar_grade(enc["batidas"], f0, bpm0, lp0, meio0, ref_alin)
    swing_m, swing_l = escolher_swing(bat_ref, f0, bpm0, lp0, meio0, ref_alin)
    if swing_m is not None:
        bat_ref = afinar_grade(bat_ref, f0, bpm0, lp0, meio0, ref_alin, swing_musica=swing_m, swing_loop=swing_l)
    batidas, fase, meio = bat_ref / a.velocidade, enc["fase"], enc["meio"]
    tempo = enc["tempo"] * a.velocidade
    compasso = 4 * (2 if meio else 1)
    compassos = batidas[fase::compasso]

    # Bateria original tratada: por baixo do loop (cola o groove) ou sozinha
    # quando nenhum loop encaixa.
    bat_cru = desacelerar(estereo["drums"], a.velocidade, a.manter_tom)[:, :n]
    bat_cru = np.pad(bat_cru, ((0, 0), (0, n - bat_cru.shape[1])))
    bat_orig = saturar(passa_baixa(bat_cru, 4500), 1.8)
    bat_a, bumbos = montar_bateria(n, batidas, fase, enc["loop_bpm"], enc["loop"], meio, swing_m, swing_l)
    # Ajuste fino: o encaixe acima mede os golpes de 11,6 em 11,6 ms e pelo
    # pico de cada um; medidos no começo da subida (passo de 1,5 ms), os
    # golpes do loop ainda chegavam 8-12 ms depois dos da bateria original
    # (e da voz, que segue a original). A grade anda esse tanto.
    d_fino = defasagem_fina(bat_a, bat_cru)
    if abs(d_fino) > 0.002:
        batidas, bat_ref = batidas - d_fino, bat_ref - d_fino * a.velocidade
        compassos = batidas[fase::compasso]
        bat_a, bumbos = montar_bateria(n, batidas, fase, enc["loop_bpm"], enc["loop"], meio, swing_m, swing_l)
    # A camada da bateria original só fica por baixo se casa com o loop; senão
    # as duas "tropeçam" uma na outra.
    _, desvio_orig, _ = defasagem_fortes_ms(bat_a, bat_orig)
    misturar_original = bool(np.isfinite(desvio_orig) and desvio_orig <= 15)
    # Loop só em música de andamento fixo (produzida no computador): um loop de
    # máquina sobre uma banda tocando ao vivo (andamento que varia, paradas,
    # balanço próprio) soa descompassado mesmo com os golpes na grade. Nesses
    # casos, substituição de bateria: cada golpe do baterista original vira
    # uma amostra lo-fi no mesmo instante.
    # Andamento fixo: pelo menos metade das batidas a menos de 25 ms de uma
    # grade fixa. Músicas que derivam um pouco entre seções ainda passam e
    # usam o loop seguindo as batidas detectadas; substituir cada ataque de
    # uma bateria trap deixaria a caixa alta e picotada. Gravações ao vivo
    # ficam em 9-15%.
    andamento_fixo = na_grade(enc["batidas"]) >= 0.5
    usar_loop = enc["nota"] >= a.encaixe_minimo and (andamento_fixo or a.bateria == "loop")
    if a.bateria in ("substituicao", "original"):
        usar_loop = False
    modo_bateria = "loop"
    if not usar_loop and a.bateria == "original":
        bat_a, bumbos = bat_orig, []
        modo_bateria = "original tratada"
    elif not usar_loop:
        from . import substituir
        golpes = substituir.classificar(mono(bat_cru))
        amostras = substituir.amostras_do_loop(librosa.load(enc["loop"], sr=SR, mono=False)[0])
        if len(golpes) >= 8 and amostras:
            # O chimbal e os pratos originais ficam (camada_pratos); só o
            # bumbo e a caixa viram amostras lo-fi.
            bat_a = substituir.substituir(n, [g for g in golpes if g[1] != "chimbal"], amostras)
            bumbos = [t for t, tipo, _ in golpes if tipo == "bumbo"]
            modo_bateria = "substituição"
        else:
            bat_a, bumbos = bat_orig, []
            modo_bateria = "original tratada"

    dur_compasso = float(np.median(np.diff(compassos))) if len(compassos) > 1 else 2.0
    secoes = [dict(ini=0.0, fim=duracao, papel="verso")]
    if variar:
        if secoes_externas:
            secoes = [dict(s) for s in secoes_externas]  # ex.: SongFormer (musicann/estrutura.py)
        else:
            secoes = arranjo.detectar_secoes(ritmo + pm["vocals"], pm["vocals"], SR)
        secoes = [dict(s, ini=s["ini"] / a.velocidade, fim=s["fim"] / a.velocidade) for s in secoes]
        secoes = arranjo.alinhar_aos_compassos(secoes, compassos, duracao)
        secoes = arranjo.juntar_curtas(secoes, max(8.0, 3 * dur_compasso))

    # Bateria: segundo loop no refrão e no instrumental.
    bateria = bat_a
    if usar_loop and variar and any(s["papel"] in ("refrao", "instrumental") for s in secoes):
        bat_b, _ = montar_bateria(n, batidas, fase, enc["loop2_bpm"], enc["loop2"], meio, swing_m, swing_l)
        no_b = arranjo.suavizar(arranjo.curva_por_papel(n, SR, secoes, dict(verso=0, refrao=1, instrumental=1)), SR, 15)
        bateria = bat_a * (1 - no_b) + nivel(bat_b, 20 * np.log10(np.sqrt(np.mean(bat_a**2)) + 1e-12)) * no_b
    # Base nova, como nos remixes lo-fi do YouTube (dados/estudo/: 93% deles
    # trocam o teclado/samples e 86% trocam o baixo): os acordes da música
    # num piano elétrico (musicann/harmonia.py) no lugar do resto original, e
    # um baixo novo colado no bumbo do loop no lugar do baixo original.
    acordes_usados, tom_usado = "", ""
    riff_toca = 0.0
    info_gancho = None
    timbre_usado = None
    toca_teclado = 0.0
    teclas_l = baixo_novo_l = None
    if a.teclado:
        from . import harmonia as hm
        timbre_usado = hm.usar_timbre(a.timbre)
        n_orig = len(pm["other"])
        lista = hm.acordes(pm["other"], pm["bass"], SR)
        tonica, modo, escala, sensivel = hm.tom_pelos_acordes(lista, hm.perfil_de_notas(pm["other"], pm["bass"], SR))
        lista = hm.acordes(pm["other"], pm["bass"], SR, escala=escala, sensivel=sensivel)
        tom_usado = f"{hm.NOMES[tonica]} {modo}"
        afinacao = float(librosa.estimate_tuning(y=pm["other"], sr=SR))
        passo = 2 if meio else 1
        grade, n_antes = hm.grade_completa(bat_ref, n_orig / SR) if andamento_fixo else (bat_ref, 0)
        groove = grade[(fase + n_antes) % (4 * passo) :: passo]
        # Os acordes entram junto com os golpes do loop, e não na grade: o
        # loop tem o próprio atraso dentro de cada batida.
        alvos = None
        tempos_musica = groove.copy()  # os tempos da música, sem o atraso do loop
        if usar_loop:
            groove = groove + atraso_na_grade(bat_a, batidas) * a.velocidade
            # Cada toque do piano que cai a menos de 35 ms de onde a bateria
            # bate vai para lá: piano e bateria batem juntos.
            alvos = posicoes_do_groove(bateria, batidas) * a.velocidade
        teclas = hm.teclado(lista, groove, n_orig, SR, afinacao, rng, escala=escala, alvos=alvos)
        toca = hm.onde_toca(lista, n_orig, SR)
        para_lofi = lambda c: arranjo.suavizar(np.interp(np.arange(n) * a.velocidade, np.arange(n_orig), c), SR, ms=300)
        if toca.any() and np.any(teclas):
            # Mesmo volume do resto original nos trechos em que o teclado toca.
            teclas *= 10 ** ((rms_db(estereo["other"][:, toca > 0]) + a.teclado_db - rms_db(teclas[:, toca > 0])) / 20)
            if a.riff:
                # Onde o rapper canta (no tempo da música): ali o piano toca
                # menos e mais baixo, respondendo à voz em vez de tocar por cima.
                rms_v = librosa.feature.rms(y=pm["vocals"], hop_length=512)[0]
                canta = arranjo.suavizar((rms_v > 0.15 * np.percentile(rms_v, 95)).astype(float), SR / 512, ms=250)
                canta = np.interp(np.arange(n_orig), np.arange(len(canta)) * 512, canta)
                # O ciclo que se repete, tocado como frase. Sem ciclo, ficam os
                # acordes (copiar o riff nota a nota soava como notas soltas).
                secoes_orig = [dict(s, ini=s["ini"] * a.velocidade, fim=s["fim"] * a.velocidade) for s in secoes]
                if a.teclado_estilo == "progressao":
                    g = hm.progressao(lista, tempos_musica, n_orig, SR, escala, afinacao, rng, alvos=alvos, canta=canta,
                                      secoes=secoes_orig, variar=a.gancho_variar)
                else:
                    g = hm.gancho(pm["other"], SR, tempos_musica, afinacao, rng, alvos=alvos, canta=canta,
                                  secoes=secoes_orig, variar=a.gancho_variar)
                rf, onde_rf = (g[0], g[1]) if g is not None else (None, np.zeros(1))
                if g is not None:
                    info_gancho = g[2]
                    nivel_db, sob_voz_db = a.gancho_db, a.gancho_sob_voz_db
                if onde_rf.any():
                    # O riff no volume do resto original onde ele toca; os
                    # acordes saem dali (com os dois juntos, o piano não para).
                    rf *= 10 ** ((rms_db(estereo["other"][:, onde_rf > 0]) + a.teclado_db - rms_db(rf[:, onde_rf > 0])) / 20)
                    teclas = teclas * (1 - arranjo.suavizar(onde_rf, SR, ms=300)) + rf
                    # E o piano recua enquanto o rapper canta e volta nas pausas.
                    teclas = teclas * (1 - (1 - 10 ** (sob_voz_db / 20)) * canta) * 10 ** (nivel_db / 20)
                    # Piso: quando o resto original é muito mais baixo que a
                    # voz, a frase no volume dele fica inaudível; ela sobe até
                    # `gancho_piso_voz_db` contra a voz.
                    fala = (canta > 0.5) & (onde_rf > 0)
                    if fala.sum() > SR:
                        rel = rms_db(teclas[:, fala]) - rms_db(estereo["vocals"][:, fala])
                        info_gancho["piano_voz_db"] = round(rel, 1)
                        if a.gancho_piso_voz_db is not None and rel < a.gancho_piso_voz_db:
                            sobe = 10 ** ((a.gancho_piso_voz_db - rel) / 20)
                            teclas = teclas * (1 + (sobe - 1) * arranjo.suavizar(onde_rf, SR, ms=300))
                            info_gancho["piano_voz_db"] = round(rms_db(teclas[:, fala]) - rms_db(estereo["vocals"][:, fala]), 1)
                    toca = np.maximum(toca, onde_rf)
                    riff_toca = float(onde_rf.mean())
            if a.cor_teclado:
                teclas = cor_do_teclado(teclas)
            teclas_l = desacelerar(teclas, a.velocidade, a.manter_tom)[:, :n]
            teclas_l = np.pad(teclas_l, ((0, 0), (0, n - teclas_l.shape[1])))
            toca_l = para_lofi(toca)
            ganho_resto = 1 - toca_l * (1 - 10 ** (a.resto_original_db / 20))
            toca_teclado = float(toca_l.mean())
            acordes_usados = " ".join(hm.nome(r, t) for i0, i1, r, t, c in lista if i1 - i0 > 0.4 and t is not None and c >= 0.6)[:200]

    # Baixo novo colado nos bumbos que o loop toca de fato (no refrão, os do
    # segundo loop). Nos tempos 1 e 3 da grade ele chegaria ~15 ms depois do
    # bumbo e, onde o loop tem outro desenho de bumbo, fora dele.
    if usar_loop and a.teclado and a.baixo_novo and acordes_usados:
        bumbos = list(ataques_finos(passa_baixa(mono(bateria), 150), 0.5))
        if bumbos:
            bn, notas_bn = hm.linha_de_baixo(lista, np.asarray(bumbos) * a.velocidade, n_orig, SR, afinacao, rng=rng)
            if notas_bn.any():
                bn *= 10 ** ((rms_db(pm["bass"][notas_bn > 0]) + a.baixo_novo_db - rms_db(bn[notas_bn > 0])) / 20)
                bn = desacelerar(bn, a.velocidade, a.manter_tom)[:n]
                baixo_novo_l = (np.stack([np.pad(bn, (0, n - len(bn)))] * 2), para_lofi(notas_bn))
    # Curvas de automação a partir do perfil de cada papel (arranjo.PERFIL).
    papel_val = lambda i: {p: arranjo.PERFIL[p][i] for p in arranjo.PERFIL}
    g_bat = arranjo.curva_por_papel(n, SR, secoes, papel_val(0))
    corte_bat = arranjo.curva_por_papel(n, SR, secoes, papel_val(1))
    corte = arranjo.curva_por_papel(n, SR, secoes, {p: v * a.corte_base for p, v in papel_val(2).items()})
    rev = arranjo.curva_por_papel(n, SR, secoes, papel_val(3))
    fronteiras = [s["ini"] for s in secoes[1:]]
    for k, s in enumerate(secoes):
        if s["papel"] == "intro":
            arranjo.rampa(corte, SR, s["ini"], s["fim"], 600, arranjo.perfil("verso")[2] * a.corte_base)
        if s["papel"] == "final":
            arranjo.rampa(g_bat, SR, s["ini"], s["ini"] + 2 * dur_compasso, 1, 0)
            g_bat[int((s["ini"] + 2 * dur_compasso) * SR):] = 0
            arranjo.rampa(corte, SR, s["ini"], s["fim"], corte[int(s["ini"] * SR)] if s["ini"] < duracao else 600, 450)
        if k + 1 < len(secoes):
            b = secoes[k + 1]["ini"]
            if secoes[k + 1]["papel"] in ("refrao", "instrumental"):
                # Antes de crescer: meio compasso sem bateria, filtro fechando e
                # abrindo de uma vez na entrada (a "subida" para o refrão).
                g_bat[int((b - dur_compasso / 2) * SR) : int(b * SR)] = 0
                arranjo.rampa(corte, SR, b - dur_compasso, b, corte[int(max(0, b - dur_compasso) * SR)], 900)
            elif s["papel"] == "refrao":
                # Saindo do refrão: última batida sem bateria (respiro).
                g_bat[int((b - dur_compasso / 4) * SR) : int(b * SR)] = 0
    g_bat, corte, corte_bat = (arranjo.suavizar(c, SR) for c in (g_bat, corte, corte_bat))
    # Voz: as mudanças de reverb entre seções deslizam em ~0,6 s, e o reverb
    # jogado na última palavra antes de cada virada entra em 0,4 s e sai em
    # 1,5 s (um corte seco soa como a voz sumindo e voltando).
    rev = np.maximum(arranjo.suavizar(rev, SR, ms=600), a.voz_sala)
    jogado = np.zeros(n)
    for fim_frase, b in zip(arranjo.fins_de_frase(mono(voz), SR, fronteiras), fronteiras):
        if fim_frase is not None:
            ini, topo, fim = max(0.0, fim_frase - 0.9), max(0.0, fim_frase - 0.5), min(duracao, b + 1.0)
            arranjo.rampa(jogado, SR, ini, topo, 0, a.reverb_jogado)
            jogado[int(topo * SR) : int(fim * SR)] = a.reverb_jogado
            arranjo.rampa(jogado, SR, fim, min(duracao, fim + 1.5), a.reverb_jogado, 0)
    rev = np.maximum(rev, arranjo.suavizar(jogado, SR, ms=150))

    # Rede de segurança: onde o loop não bate com a bateria original, ele sai.
    trechos_mudos = 0
    if usar_loop:
        seguro, trechos_mudos = rede_de_seguranca(bateria, bat_orig, duracao, n)
        g_bat = g_bat * arranjo.suavizar(seguro, SR, ms=300)
    # Onde a bateria nova não toca (introdução, ponte, viradas, final, trechos
    # tirados pela rede de segurança), a bateria original entra junto com a
    # base, com o mesmo filtro. O Demucs separa a música em partes que só soam
    # limpas somadas: o baixo sozinho tem "buracos" onde a separação deu a
    # energia do bumbo para a bateria, e sem a bateria por cima isso soava como
    # interferência, o som cortando. Com as partes filtradas juntas, é como
    # filtrar a própria música: a introdução fica a música abafada, abrindo.
    orig_na_base = arranjo.suavizar(1 - np.clip(g_bat, 0, 1), SR, ms=150)
    if teclas_l is not None:
        harmonia = baixo + resto * ganho_resto + teclas_l
    if baixo_novo_l is not None:
        # O baixo novo só entra onde o loop toca (é colado no bumbo dele).
        bn, notas_bn = baixo_novo_l
        uso = notas_bn * arranjo.suavizar((g_bat > 0.5).astype(float), SR, ms=200)
        harmonia = harmonia - baixo * uso * (1 - 10 ** (a.baixo_original_db / 20)) + bn * uso
    harmonia = harmonia + bat_cru * orig_na_base + assobio

    base = passa_alta(harmonia, 35)
    base = np.stack([arranjo.passa_baixa_variavel(c, corte, SR) for c in base])
    base = saturar(base, a.saturacao)
    bumbos = [b for b in bumbos if b < duracao and g_bat[min(n - 1, int(b * SR))] > 0.5]
    base = sidechain(base, bumbos, a.sidechain)

    # Voz: continua clara, mas com um toque de fita o tempo todo (agudos um
    # pouco mais macios, saturação leve e um mínimo de sala), para não soar
    # idêntica à original nos trechos em que ela canta quase sozinha.
    voz = passa_alta(voz, 90)
    voz = menos_agudos(voz, a.voz_agudos_hz, a.voz_agudos_db)
    voz = saturar(voz, a.voz_saturacao)
    molhada = reverb(voz, 1.0, rng=rng)
    voz = (1 - rev) * voz + rev * molhada

    # A bateria passa pela mesma "fita" da base (filtro, saturação, oscilação
    # e um pouco de sala), para soar gravada junto e não colada por cima.
    if usar_loop and misturar_original:
        rms_loop = np.sqrt(np.mean(bateria**2)) + 1e-12
        bateria = bateria + nivel(bat_orig, 20 * np.log10(rms_loop) + a.bateria_original_db)
    bateria = np.stack([arranjo.passa_baixa_variavel(c, corte_bat, SR) for c in bateria])
    bateria = saturar(bateria, 1.6)
    bateria = reverb(bateria, 0.12, rt60=0.6, rng=rng)
    bateria = limitar(bateria) * g_bat
    r_voz = rms_db(estereo["vocals"])
    vol_bat = a.volume_voz + float(np.clip(rms_db(estereo["drums"]) - r_voz + a.mais_que_original,
                                           a.volume_bateria - a.volume_voz, a.teto_bateria))
    # O teto da base deixa a voz na frente, mas sem tirar mais que
    # `perda_maxima_db` do equilíbrio da original: num trap com 808 alto, o
    # teto fixo poria a voz muito mais à frente do que na própria original.
    base_orig = rms_db(estereo["bass"] + estereo["other"]) - r_voz
    vol_base = a.volume_voz + float(np.clip(base_orig + a.mais_que_original, a.volume_base - a.volume_voz,
                                            max(a.teto_base, base_orig - a.perda_maxima_db)))
    nv_bat = nivel(bateria, vol_bat) if np.any(bateria) else bateria
    # Pratos e chimbal da bateria original, no volume que tinham em relação à
    # voz na música original (menos `pratos_db`), onde a bateria nova toca.
    topo_cru = passa_alta(bat_cru, a.pratos_hz, ordem=4)
    rel = rms_db(topo_cru) - rms_db(desacelerar(estereo["vocals"], a.velocidade, a.manter_tom))
    pratos = nivel(camada_pratos(bat_cru, a.pratos_hz), a.volume_voz + rel + a.pratos_db) * g_bat
    mix = nivel(voz, a.volume_voz) + nivel(base, vol_base) + nv_bat + pratos
    if camadas is not None:
        camadas.update(voz=voz, bateria=nv_bat, pratos=pratos, bateria_original=bat_cru, grade=batidas,
                       bumbos=np.asarray(bumbos), teclado=teclas_l,
                       baixo_novo=None if baixo_novo_l is None else baixo_novo_l[0] * uso,
                       # ganho (dB) que a base e a voz levam na mixagem
                       ganho_base_db=vol_base - rms_db(base), ganho_voz_db=a.volume_voz - rms_db(voz))
    # A oscilação de fita vai na mixagem inteira de uma vez: voz e base
    # oscilam juntas (separadas, elas desafinavam uma contra a outra).
    mix = wow_flutter(mix, a.wow, rng)
    mix = menos_graves(mix, a.corte_graves, a.graves_db)
    mix = passa_baixa(mix, a.corte_final, ordem=2)
    mix = mix + vinil(n, a.chiado, a.estalos, rng)
    # Final: some aos poucos nos últimos 4 s.
    f = min(n, int(4 * SR))
    mix[:, -f:] *= np.linspace(1, 0, f) ** 1.5
    info = dict(tempo=round(float(tempo), 1), meio_tempo=bool(meio), encaixe=round(enc["nota"], 2),
                loop=os.path.basename(enc["loop"]) if usar_loop else "bateria original",
                loop2=os.path.basename(enc["loop2"]), bateria=modo_bateria, swing=swing_m if usar_loop else None, bateria_original_por_baixo=misturar_original and usar_loop, trechos_sem_bateria=trechos_mudos,
                alinhamento_ms=round(float(desvio_orig), 1) if usar_loop else None,
                teclado=round(toca_teclado, 2), riff=round(riff_toca, 2), gancho=info_gancho, timbre=timbre_usado, tom_da_musica=tom_usado, acordes=acordes_usados,
                baixo_novo=baixo_novo_l is not None, assobio_na_base=tem_assobio, bateria_db=round(vol_bat - a.volume_voz, 1), base_db=round(vol_base - a.volume_voz, 1), secoes=[(round(s["ini"]), s["papel"]) for s in secoes])
    # Saída (n, 2), pronta para soundfile.write.
    return (0.9 * mix / (np.abs(mix).max() + 1e-9)).T.astype(np.float32), info


# ---------- precisão do encaixe ----------

def defasagem_ms(bateria_nova, ritmo_original, janela=0.1):
    """Atraso de cada golpe da bateria nova em relação ao ataque mais próximo
    da música original (ms; positivo = bateria nova atrasada). Devolve
    (mediana com sinal, mediana absoluta, fração de golpes a mais de 30 ms)."""
    novos = librosa.onset.onset_detect(y=mono(bateria_nova), sr=SR, units="time", backtrack=True)
    orig = librosa.onset.onset_detect(y=mono(ritmo_original), sr=SR, units="time", backtrack=True)
    if len(novos) == 0 or len(orig) == 0:
        return np.nan, np.nan, np.nan
    idx = np.clip(np.searchsorted(orig, novos), 1, len(orig) - 1)
    perto = np.where(np.abs(orig[idx - 1] - novos) < np.abs(orig[idx] - novos), orig[idx - 1], orig[idx])
    d = (novos - perto)
    d = d[np.abs(d) < janela] * 1000
    return float(np.median(d)), float(np.median(np.abs(d))), float(np.mean(np.abs(d) > 30))


def ataques_finos(y, fracao=0.5):
    """Instantes (s) dos `fracao` ataques mais fortes, no começo da subida
    de cada um (passo de 64 amostras, 1,5 ms)."""
    hop = 64
    env = librosa.onset.onset_strength(y=mono(y).astype(np.float32), sr=SR, hop_length=hop)
    q = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, backtrack=True)
    if len(q) == 0:
        return np.zeros(0)
    forca = env[np.minimum(q + 3, len(env) - 1)]
    q = q[forca >= np.quantile(forca, 1 - fracao)]
    return librosa.frames_to_time(q, sr=SR, hop_length=hop)


def defasagem_fina(nova, original, maximo=0.04):
    """Atraso mediano (s) dos ataques fortes de `nova` em relação aos de
    `original` (só os que têm um par a menos de `maximo` s)."""
    a, r = ataques_finos(nova), ataques_finos(original)
    if len(a) < 8 or len(r) < 8:
        return 0.0
    _, d = _atrasos(a, np.sort(r), janela=maximo)
    return float(np.median(d)) if len(d) >= 8 else 0.0


def atraso_na_grade(bateria, batidas, maximo=0.04):
    """Onde os golpes fortes da bateria caem em relação às batidas da grade
    (mediana, s), contando só os que estão a menos de `maximo` s de uma."""
    a = ataques_finos(bateria)
    if len(a) < 8 or len(batidas) < 2:
        return 0.0
    _, d = _atrasos(a, np.asarray(batidas), janela=maximo)
    return float(np.median(d)) if len(d) >= 8 else 0.0


def posicoes_do_groove(bateria, batidas, fracao=0.7, minimo=0.15):
    """Onde a bateria bate (s): os golpes de fato e, em cada batida da grade,
    as posições em que o loop costuma bater (as que aparecem em >= `minimo`
    das batidas). Quando o loop tem swing, as notas do piano copiadas do
    teclado original caem ora no contratempo reto, ora no com swing; levadas
    a essas posições, seguem o balanço da bateria."""
    golpes = ataques_finos(bateria, fracao)
    b = np.asarray(batidas)
    if len(golpes) < 8 or len(b) < 4:
        return golpes
    j = np.searchsorted(b, golpes) - 1
    ok = (j >= 0) & (j < len(b) - 1)
    fase = (golpes[ok] - b[j[ok]]) / (b[j[ok] + 1] - b[j[ok]])
    caixas = np.histogram(fase, bins=48, range=(0, 1))[0]
    fases = []
    for k in np.flatnonzero(caixas >= minimo * len(b)):
        if fases and k - fases[-1][1] <= 1:
            fases[-1][1] = k
        else:
            fases.append([k, k])
    tipicas = [float(np.median(fase[(fase >= k0 / 48) & (fase < (k1 + 1) / 48)])) for k0, k1 in fases]
    virtuais = [b[i] + f * (b[i + 1] - b[i]) for i in range(len(b) - 1) for f in tipicas]
    return np.sort(np.concatenate([golpes, virtuais]))


def golpes_fortes(y, fracao=0.4):
    """Instantes (s) dos golpes mais fortes (bumbo e caixa): os `fracao`
    ataques de maior intensidade. São eles que o ouvido usa para sentir o
    tempo; chimbal e violão são numerosos e puxariam qualquer média."""
    env = librosa.onset.onset_strength(y=mono(y), sr=SR, hop_length=HOP_ENC)
    q = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP_ENC, units="frames")
    if len(q) == 0:
        return np.zeros(0)
    q = q[env[q] >= np.quantile(env[q], 1 - fracao)]
    return librosa.frames_to_time(q, sr=SR, hop_length=HOP_ENC)


def _atrasos(novos, referencia, janela=0.1):
    idx = np.clip(np.searchsorted(referencia, novos), 1, len(referencia) - 1)
    perto = np.where(np.abs(referencia[idx - 1] - novos) < np.abs(referencia[idx] - novos), referencia[idx - 1], referencia[idx])
    d = novos - perto
    ok = np.abs(d) < janela
    return novos[ok], d[ok]


def defasagem_fortes_ms(bateria_nova, bateria_original):
    """Atraso dos golpes fortes da bateria nova em relação aos golpes fortes
    da bateria original: (mediana com sinal, mediana absoluta, fração > 30 ms)."""
    ref = golpes_fortes(bateria_original, 0.5)
    nov = golpes_fortes(bateria_nova, 0.4)
    if len(ref) < 4 or len(nov) < 4:
        return np.nan, np.nan, np.nan
    _, d = _atrasos(nov, ref)
    d = d * 1000
    return float(np.median(d)), float(np.median(np.abs(d))), float(np.mean(np.abs(d) > 30))


def afinar_grade(batidas, fase, loop_bpm, loop, meio, bateria_original, iteracoes=4, janela_s=8.0,
                 swing_musica=None, swing_loop=0.5):
    """Corrige a grade golpe a golpe: alinha os golpes fortes do loop (bumbo e
    caixa) com os golpes fortes da bateria original da música.

    Mede o atraso de cada golpe forte em relação ao golpe forte mais próximo
    da bateria original e desloca a grade pela mediana dos atrasos num trecho
    de `janela_s` segundos em volta de cada batida. Tira o atraso fixo da
    grade do detector e acompanha a deriva de gravações tocadas ao vivo."""
    if isinstance(loop, str):
        loop, _ = librosa.load(loop, sr=SR, mono=False)
    n = len(mono(bateria_original))
    ref = golpes_fortes(bateria_original, 0.5)
    if len(ref) < 8:
        return batidas
    for _ in range(iteracoes):
        tr, _ = montar_bateria(n, batidas, fase, loop_bpm, loop, meio, swing_musica, swing_loop)
        nov = golpes_fortes(tr, 0.4)
        t, d = _atrasos(nov, ref)
        if len(d) < 8:
            break
        geral = np.median(d)
        corr = np.array([np.median(d[np.abs(t - b) < janela_s / 2]) if np.sum(np.abs(t - b) < janela_s / 2) >= 3 else geral
                         for b in batidas])
        if np.max(np.abs(corr)) < 0.003:
            break
        batidas = batidas - corr
    return batidas


def escolher_swing(batidas, fase, loop_bpm, loop, meio, ritmo_original, candidatos=(0.5, 0.54, 0.58, 0.62, 0.66)):
    """Testa valores de swing e fica com o que deixa os golpes da bateria nova
    mais perto dos golpes fortes de `ritmo_original` (a bateria original).
    Estimar o swing pela posição dos ataques erra quando a música tem notas
    mais rápidas (semicolcheias) misturadas aos contratempos."""
    if isinstance(loop, str):
        loop, _ = librosa.load(loop, sr=SR, mono=False)
    swing_loop = swing_do_loop(loop, loop_bpm)
    if meio:
        return None, swing_loop
    n = len(ritmo_original)
    melhor, erro_min = None, np.inf
    for s in candidatos:
        tr, _ = montar_bateria(n, batidas, fase, loop_bpm, loop, meio, s, swing_loop)
        _, erro, _ = defasagem_fortes_ms(tr, ritmo_original)
        if erro < erro_min - 0.5:
            melhor, erro_min = s, erro
    return melhor, swing_loop
