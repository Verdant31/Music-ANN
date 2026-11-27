"""Teclado novo: os acordes da música tocados num piano elétrico lo-fi.

As lo-fi mais ouvidas do YouTube (Yuko Records, jdutra, Smooth Sounds...)
são remixes: o produtor guarda a voz e refaz a base, com acordes "jazzy"
(com sétima e nona) num piano elétrico tipo Rhodes. Manter o instrumental
original, só filtrado, deixa a lo-fi parecida demais com a música.

  acordes()   acha os acordes da música no baixo e no resto (chroma contra
              moldes de acordes, suavizado com Viterbi), com uma confiança
  vozes()     escolhe as notas de cada acorde (com sétima e nona, perto
              das notas do acorde anterior, como um tecladista tocaria)
  rhodes()    sintetiza uma nota de piano elétrico (modelo físico simples;
              com as amostras do jRhodes3d, toca o Rhodes gravado)
  teclado()   toca os acordes na grade de compassos, com o ritmo e as
              pequenas imperfeições de um tecladista
"""
from collections import Counter

import librosa
import numpy as np
from scipy import signal

NOMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Intervalos (semitons) de cada tipo de acorde e o peso de cada nota no molde.
TIPOS = {
    "maj": ((0, 4, 7), (1.0, 0.8, 0.8)),
    "min": ((0, 3, 7), (1.0, 0.8, 0.8)),
    "7": ((0, 4, 7, 10), (1.0, 0.8, 0.7, 0.6)),
    "maj7": ((0, 4, 7, 11), (1.0, 0.8, 0.7, 0.6)),
    "min7": ((0, 3, 7, 10), (1.0, 0.8, 0.7, 0.6)),
}


# Perfis de Krumhansl-Kessler (quanto cada grau "pertence" ao tom).
PERFIL_MAIOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# Notas de cada tipo de acorde (com o "5": sem terça, para quando não dá
# para saber se o acorde é maior ou menor).
INTERVALOS = {**{t: iv for t, (iv, _) in TIPOS.items()}, "5": (0, 7), "6": (0, 4, 7)}


def tonalidade(harmonia, baixo, sr, hop=4096):
    """Tom da música: (tônica 0-11, "maior" ou "menor", notas da escala).
    Na escala menor entra também a sensível (o acorde de dominante maior)."""
    c = librosa.feature.chroma_cqt(y=harmonia + baixo, sr=sr, hop_length=hop)
    nivel = librosa.feature.rms(y=harmonia + baixo, frame_length=2 * hop, hop_length=hop)[0][: c.shape[1]]
    media = (c * nivel[None, : c.shape[1]]).sum(1)
    notas = []
    for t in range(12):
        notas.append((np.corrcoef(media, np.roll(PERFIL_MAIOR, t))[0, 1], t, "maior"))
        notas.append((np.corrcoef(media, np.roll(PERFIL_MENOR, t))[0, 1], t, "menor"))
    _, tonica, modo = max(notas)
    graus = (0, 2, 4, 5, 7, 9, 11) if modo == "maior" else (0, 2, 3, 5, 7, 8, 10, 11)
    return tonica, modo, {(tonica + g) % 12 for g in graus}


def _moldes():
    rotulos, M = [], []
    for raiz in range(12):
        for tipo, (iv, peso) in TIPOS.items():
            v = np.zeros(12)
            for i, p in zip(iv, peso):
                v[(raiz + i) % 12] = p
            rotulos.append((raiz, tipo))
            M.append(v / np.linalg.norm(v))
    return rotulos, np.array(M)


def _setima_menor_clara(v, raiz):
    """A sétima menor soa claramente (bem mais que a maior e com força perto
    das notas do acorde)? v: chroma médio do trecho."""
    b7, m7 = v[(raiz + 10) % 12], v[(raiz + 11) % 12]
    acorde = np.mean([v[(raiz + i) % 12] for i in (0, 4, 7)])
    return b7 > 1.5 * m7 and b7 > 0.4 * acorde


def acordes(harmonia, baixo, sr, hop=4096, troca=0.995, escala=None, sensivel=None, duvida=0.02, terca_clara=0.03):
    """Acordes da música: lista de (ini s, fim s, raiz 0-11, tipo, confiança).

    harmonia: o resto (teclados, guitarras, samples); baixo: o baixo. O
    chroma do baixo conta em dobro para a raiz (o baixo quase sempre toca a
    raiz no tempo forte). A confiança é a semelhança média do chroma com o
    molde escolhido; trechos sem harmonia (só voz, silêncio) saem com tipo
    None. Com `escala`, acordes com notas fora do tom perdem pontos (só
    ganham com evidência forte): o chroma confunde, por exemplo, Fá menor
    com Fá maior quando o lá bemol da melodia está fraco."""
    rotulos, M = _moldes()
    c = librosa.feature.chroma_cqt(y=harmonia + baixo, sr=sr, hop_length=hop)
    cb = librosa.feature.chroma_cqt(y=baixo, sr=sr, hop_length=hop, fmin=librosa.note_to_hz("C1"), n_octaves=4)
    n = min(c.shape[1], cb.shape[1])
    c, cb = c[:, :n], cb[:, :n]
    nivel = librosa.feature.rms(y=harmonia + baixo, frame_length=2 * hop, hop_length=hop)[0][:n]
    calado = nivel < 0.1 * np.percentile(nivel, 90)
    cn = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)
    sim = M @ cn  # (moldes, quadros)
    # Só o resto, sem o baixo: para desempatar a terça. Um 808 forte domina
    # o chroma da soma (e o 5º harmônico dele cai na terça maior), e o acorde
    # menor da melodia sairia sem terça.
    ch = librosa.feature.chroma_cqt(y=harmonia, sr=sr, hop_length=hop)[:, :n]
    sim_h = M @ (ch / (np.linalg.norm(ch, axis=0, keepdims=True) + 1e-9))
    raiz_baixo = cb / (cb.max(0, keepdims=True) + 1e-9)
    bonus = np.array([raiz_baixo[r] for r, _ in rotulos]) * 0.25
    # Tríade simples tem uma leve preferência (sétima só quando ela está lá).
    simples = np.array([0.03 if t in ("maj", "min") else 0 for _, t in rotulos])[:, None]
    fora = np.zeros((len(rotulos), 1))
    if escala:
        fora[:, 0] = [-0.06 * fora_do_tom(r, t, escala, sensivel) for r, t in rotulos]
    nota = sim + bonus + simples + fora
    # Viterbi: ficar no mesmo acorde é muito mais provável que trocar.
    K = len(rotulos)
    log_t = np.full((K, K), np.log((1 - troca) / (K - 1)))
    np.fill_diagonal(log_t, np.log(troca))
    emis = 8 * nota  # "temperatura" da observação
    caminho = np.zeros((K, n), int)
    v = emis[:, 0].copy()
    for t in range(1, n):
        cand = v[:, None] + log_t
        caminho[:, t] = np.argmax(cand, 0)
        v = cand[caminho[:, t], np.arange(K)] + emis[:, t]
    seq = np.zeros(n, int)
    seq[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        seq[t - 1] = caminho[seq[t], t]
    tempos = librosa.frames_to_time(np.arange(n + 1), sr=sr, hop_length=hop)
    saida = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        raiz, tipo = rotulos[seq[i]]
        conf = float(sim[seq[i], i : j + 1].mean())
        # Terça em dúvida (o acorde paralelo, maior <-> menor, quase empata):
        # o teclado toca sem terça, que combina com os dois.
        paralelo = {"maj": "min", "min": "maj", "maj7": "min7", "min7": "maj7", "7": "min7"}[tipo]
        k_par = rotulos.index((raiz, paralelo))
        margem = float((nota[seq[i], i : j + 1] - nota[k_par, i : j + 1]).mean())
        if margem < duvida:
            # Na soma a terça empata; o resto sozinho (sem o baixo) desempata
            # quando é claro. Se nem ele decide, o teclado toca sem terça.
            margem_h = float((sim_h[seq[i], i : j + 1] - sim_h[k_par, i : j + 1]).mean())
            if margem_h <= -terca_clara:
                tipo = paralelo
            elif margem_h < terca_clara:
                tipo = "5"
        elif tipo == "maj" and _setima_menor_clara(c[:, i : j + 1].mean(1), raiz):
            # Acorde maior em que soa mais a sétima menor que a maior (rock,
            # blues, soul): o teclado toca 6/9 em vez de sétima maior.
            tipo = "6"
        if calado[i : j + 1].mean() > 0.5:
            tipo = None
        saida.append((float(tempos[i]), float(tempos[j + 1]), raiz, tipo, conf))
        i = j + 1
    return saida


def nome(raiz, tipo):
    return "N" if tipo is None else NOMES[raiz] + {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "5": "5", "6": "6"}[tipo]


def notas_lofi(raiz, tipo, escala=None):
    """Intervalos do acorde de lo-fi: o acorde detectado com sétima e nona,
    mas só as que pertencem ao tom (uma nona fora do tom, como o lá natural
    num Sol menor dentro de Mi bemol maior, briga com a melodia)."""
    no_tom = lambda i: escala is None or (raiz + i) % 12 in escala
    iv = [0] if tipo == "5" else [0, 3 if tipo in ("min", "min7") else 4]
    if tipo != "7":
        iv.append(7)
    if tipo == "7":
        setima = 10
    elif tipo == "maj7":
        setima = 11
    elif tipo in ("min7", "min", "5"):
        setima = 10 if no_tom(10) else None
    elif tipo == "maj" and no_tom(11):
        setima = 11
    else:
        # "6" (acorde maior com sétima menor soando) ou sétima maior fora do
        # tom: sexta no lugar da sétima (6/9), que não briga com a sétima
        # menor dos acordes de rock e blues.
        setima = 9 if no_tom(9) else None
    if setima is not None:
        iv.append(setima)
    if no_tom(14):
        iv.append(14)
    elif tipo == "7" and no_tom(21):
        iv.append(21)
    return iv


def vozes(raiz, tipo, anterior=None, escala=None, centro=60):
    """Notas MIDI do acorde de lo-fi: raiz no grave (em volta de C3) e as
    outras notas perto das do acorde anterior (condução de vozes)."""
    iv = notas_lofi(raiz, tipo, escala)
    baixo = 48 + (raiz - 48) % 12 - (12 if (48 + (raiz - 48) % 12) > 52 else 0)
    superiores = [raiz + i for i in iv[1:]]
    alvo = np.mean(anterior[1:]) if anterior else centro + 4
    notas = []
    for s_ in superiores:
        opcoes = [s_ % 12 + 12 * o for o in range(4, 7)]
        notas.append(min(opcoes, key=lambda x: abs(x - alvo)))
    return [baixo] + sorted(set(notas))


def rhodes(f, dur, sr, vel=0.8, rng=None):
    """Uma nota de piano elétrico (Rhodes) pelo que soa num de verdade: a
    lâmina vibrando quase em seno, o "sino" do primeiro modo de flexão dela
    (~6,3 vezes a frequência, inarmônico, some em dezenas de ms), o captador
    magnético, que não é linear (assimétrico: soma 2º harmônico, o "rosnado",
    mais forte quanto mais forte a nota e mais no ataque), a batidinha do
    martelo de feltro, e o brilho que cresce com a força e cai com a altura.
    Decai em dois estágios (rápido, depois longo; as graves soam mais) e,
    ao soltar, o abafador corta em ~70 ms.

    Substitui o modelo FM (rhodes_fm), em que no ataque o 2º e o 3º
    harmônicos iam tão fortes quanto a fundamental e, em meio segundo, a
    nota virava quase seno puro (o timbre fino de sintetizador)."""
    rng = rng or np.random.default_rng(0)
    k = (261.6 / f) ** 0.5  # graves soam mais e mais escuras
    n = int((dur + 0.4) * sr)
    t = np.arange(n) / sr
    amp = (0.5 * np.exp(-t / (0.3 * k + 0.08)) + 0.5 * np.exp(-t / (2.0 * k + 0.5))) * (1 - np.exp(-t / 0.002))
    amp *= np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.07))
    fase = rng.uniform(0, 2 * np.pi)
    x = np.sin(2 * np.pi * f * t + fase)
    fb = 6.27 * f
    if fb < 0.45 * sr:
        sino = (0.05 + 0.2 * vel) * np.clip(1.4 - f / 900, 0.2, 1.0)
        x = x + sino * np.exp(-t / (0.02 + 0.03 * k)) * np.sin(2 * np.pi * fb * t + rng.uniform(0, 2 * np.pi))
    # tonebar: um pouco de 2º harmônico que fica (o corpo quente)
    x = x + 0.08 * np.sin(4 * np.pi * f * t + 2 * fase)
    x *= amp
    c2 = 0.25 + 0.55 * vel
    y = x + c2 * x ** 2 - 0.1 * x ** 3
    y -= signal.lfilter([0.002], [1, -0.998], c2 * x ** 2)  # tira o DC que o x² cria
    m = int(0.006 * sr)
    martelo = rng.normal(0, 1, m) * np.exp(-np.arange(m) / (0.0015 * sr)) * 0.04 * vel
    y[:m] += signal.lfilter([0.15], [1, -0.85], martelo)
    fc = min(0.45 * sr, max(2.5 * f, (1200 + 4200 * vel) * np.clip(1.3 - f / 2000, 0.5, 1.2)))
    a = np.exp(-2 * np.pi * fc / sr)
    y = signal.lfilter([1 - a], [1, -a], y)
    return 0.8 * (vel ** 1.5) * y


# --- Rhodes gravado (amostras) ---------------------------------------------
# Cada nota é uma amostra gravada de um Rhodes Mark I Stage 73 de 1977
# (jRhodes3d, de Jeff Learman: https://github.com/sfzinstruments/jlearman.jRhodes3d,
# amostras CC BY 4.0, livres para fazer música). As amostras ficam em
# dados/instrumentos (fora do git); sem elas, toca o sintetizado.
INSTRUMENTO = "dados/instrumentos/jlearman.jRhodes3d-master/jRhodes3d-mono/_jRhodes3d-mono-no-xfade-flac.sfz"
_TIMBRE = {"nome": "sintetizado", "regioes": None, "cache": {}}


def _nota_sfz(nome_nota):
    """Nome de nota do SFZ (C4 = 60; aceita bemol 'b' e sustenido '#') para MIDI."""
    if nome_nota.lstrip("-").isdigit():
        return int(nome_nota)
    base = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[nome_nota[0].lower()]
    resto = nome_nota[1:]
    while resto and resto[0] in "b#":
        base += -1 if resto[0] == "b" else 1
        resto = resto[1:]
    return base + 12 * (int(resto) + 1)


def ler_sfz(caminho):
    """Regiões de um SFZ simples (<group>/<region> com sample, lokey, hikey,
    pitch_keycenter, lovel, hivel, ampeg_release): o que o jRhodes3d usa."""
    import os
    regioes, grupo, pasta = [], {}, os.path.dirname(caminho)
    for linha in open(caminho, encoding="utf-8", errors="replace"):
        linha = linha.split("//")[0].strip()
        if not linha:
            continue
        for bloco in linha.replace("<", "\n<").split("\n"):
            bloco = bloco.strip()
            if not bloco:
                continue
            cab = bloco.split(">")[0] + ">" if bloco.startswith("<") else ""
            pares = dict(p.split("=", 1) for p in bloco[len(cab):].split() if "=" in p)
            if cab == "<group>":
                grupo = pares
            elif cab == "<region>":
                r = dict(grupo, **pares)
                regioes.append(dict(amostra=os.path.join(pasta, r["sample"]),
                                    lokey=_nota_sfz(r.get("lokey", "0")), hikey=_nota_sfz(r.get("hikey", "127")),
                                    centro=_nota_sfz(r.get("pitch_keycenter", r.get("lokey", "60"))),
                                    lovel=int(r.get("lovel", 1)), hivel=int(r.get("hivel", 127)),
                                    solta=float(r.get("ampeg_release", 0.3))))
    return regioes


def usar_timbre(nome):
    """'gravado' (amostras do jRhodes3d, se existirem) ou 'sintetizado'."""
    import os
    _TIMBRE["nome"] = "sintetizado"
    if nome == "gravado" and os.path.exists(INSTRUMENTO):
        if _TIMBRE["regioes"] is None:
            _TIMBRE["regioes"] = ler_sfz(INSTRUMENTO)
        _TIMBRE["nome"] = "gravado"
    return _TIMBRE["nome"]


def rhodes_gravado(f, dur, sr, vel=0.8, rng=None):
    """Uma nota do Rhodes gravado: a amostra da camada de força e da região
    da nota, afinada por reamostragem (no máximo ~2 semitons, porque há uma
    amostra a cada 4 teclas brancas), soando `dur` s e soltando no tempo de
    abafador do instrumento. Gasta os mesmos sorteios de `rng` que rhodes(),
    para as duas versões terem exatamente os mesmos toques (força e tempo)."""
    import soundfile as sf
    rng = rng or np.random.default_rng(0)
    rng.uniform(0, 2 * np.pi)
    if 6.27 * f < 0.45 * sr:
        rng.uniform(0, 2 * np.pi)
    rng.normal(0, 1, int(0.006 * sr))
    m = float(librosa.hz_to_midi(f))
    v = int(np.clip(round(vel * 127), 1, 127))
    tecla = int(round(m))
    opcoes = [r for r in _TIMBRE["regioes"] if r["lokey"] <= tecla <= r["hikey"]]
    if not opcoes:
        opcoes = sorted(_TIMBRE["regioes"], key=lambda r: abs(r["centro"] - tecla))[:5]
    r = min(opcoes, key=lambda r: 0 if r["lovel"] <= v <= r["hivel"] else min(abs(v - r["lovel"]), abs(v - r["hivel"])))
    if r["amostra"] not in _TIMBRE["cache"]:
        y, sr_a = sf.read(r["amostra"], dtype="float32")
        y = y.mean(1) if y.ndim == 2 else y
        if sr_a != sr:
            y = librosa.resample(y, orig_sr=sr_a, target_sr=sr)
        _TIMBRE["cache"][r["amostra"]] = y
    y = _TIMBRE["cache"][r["amostra"]]
    razao = 2 ** ((m - r["centro"]) / 12)
    n = int((dur + r["solta"]) * sr)
    pos = np.arange(n) * razao
    pos = pos[pos < len(y) - 1]
    saida = np.interp(pos, np.arange(len(y)), y)
    t = np.arange(len(saida)) / sr
    saida *= np.where(t < dur, 1.0, 10 ** (-3 * (t - dur) / r["solta"]))  # -60 dB no tempo de soltar
    return 0.8 * (vel ** 1.5) * saida


def piano(f, dur, sr, vel=0.8, rng=None):
    """A nota do piano elétrico no timbre escolhido (usar_timbre)."""
    if _TIMBRE["nome"] == "gravado":
        return rhodes_gravado(f, dur, sr, vel, rng)
    return rhodes(f, dur, sr, vel, rng)


def rhodes_fm(f, dur, sr, vel=0.8, rng=None):
    """O piano elétrico antigo (FM com o modulador na mesma frequência e um
    "plim" no ataque), guardado para comparação. Ver rhodes."""
    rng = rng or np.random.default_rng(0)
    n = int((dur + 0.25) * sr)
    t = np.arange(n) / sr
    decai = 1.2 * (261.6 / f) ** 0.4 + 0.6  # notas graves soam mais
    amp = np.exp(-t / decai) * (1 - np.exp(-t / 0.003))
    solta = np.clip((dur + 0.25 - t) / 0.25, 0, 1)
    amp *= np.where(t < dur, 1.0, solta)
    indice = (0.6 + 1.4 * vel) * np.exp(-t / 0.35) + 0.25
    lamina = 0.35 * vel * np.exp(-t / 0.04) * np.sin(2 * np.pi * 14 * f * t)
    fase = rng.uniform(0, 2 * np.pi)
    y = np.sin(2 * np.pi * f * t + fase + indice * np.sin(2 * np.pi * f * t) + lamina)
    return (vel ** 1.5) * amp * y


def no_golpe(t, alvos, raio=0.035):
    """`t` levado ao golpe de `alvos` mais próximo, se houver um a menos de `raio` s."""
    if alvos is None or len(alvos) == 0:
        return t
    j = int(np.argmin(np.abs(alvos - t)))
    return float(alvos[j]) if abs(alvos[j] - t) <= raio else t


def _acorde_na_batida(lista, t0, t1):
    """O acorde que mais soa entre t0 e t1 (ou None)."""
    melhor, tempo = None, 0.0
    for a in lista:
        sob = min(a[1], t1) - max(a[0], t0)
        if sob > tempo:
            melhor, tempo = a, sob
    return melhor


def teclado(lista_acordes, batidas, n, sr, afinacao=0.0, rng=None, conf_minima=0.6, escala=None, alvos=None):
    """Faixa estéreo (2, n) com os acordes tocados no piano elétrico.

    batidas: os tempos do groove (s), o primeiro abrindo um compasso de 4.
    Em cada compasso: acorde cheio no tempo 1 e um toque mais leve no
    contratempo do 2 (o padrão mais comum de teclado em lo-fi); se o acorde
    da música muda em outro tempo, toca a troca ali. Dedilhado de baixo para
    cima em ~7 ms por nota, com o tempo e a força um pouco irregulares (com
    15 ms por nota e ±6 ms de sorteio, a última nota chegava ~50 ms depois
    do golpe, e o piano soava fora do tempo). Acordes com
    confiança baixa ou sem harmonia ficam de fora (melhor faltar teclado que
    tocar a nota errada por cima da voz). afinacao: desvio da música em
    relação ao lá 440 (fração de semitom, librosa.estimate_tuning).
    alvos: golpes da bateria (s); o toque que cai perto de um vai para ele."""
    rng = rng or np.random.default_rng(0)
    toques = []  # (início, acorde, força)
    atual = None
    for i in range(len(batidas) - 1):
        t0, t1 = batidas[i], batidas[i + 1]
        a = _acorde_na_batida(lista_acordes, t0, t1)
        ok = a is not None and a[3] is not None and a[4] >= conf_minima
        chave = (a[2], a[3]) if ok else None
        pos = i % 4
        if chave is None:
            if atual is not None:
                toques.append((t0, None, 0))
            atual = None
            continue
        if chave != atual or pos == 0:
            toques.append((t0, a, 0.85 if pos == 0 else 0.7))
        elif pos == 1:
            toques.append((t0 + 0.5 * (t1 - t0), a, 0.45))
        atual = chave
    if batidas is not None and len(batidas):
        toques.append((batidas[-1], None, 0))
    toques = [(no_golpe(t, alvos), ac, v) for t, ac, v in toques]
    esq, dir_ = np.zeros(n), np.zeros(n)
    anterior, ultimo = None, None
    for k, (ini, a, vel) in enumerate(toques[:-1]):
        if a is None:
            continue
        dur = toques[k + 1][0] - ini
        chave = (a[2], a[3])
        notas = vozes(a[2], a[3], anterior, escala) if chave != ultimo or anterior is None else anterior
        anterior, ultimo = notas, chave
        for j, m in enumerate(notas):
            ti = ini + j * 0.007 + rng.normal(0, 0.003)
            v = vel * rng.uniform(0.85, 1.0) * (0.8 if j == 0 else 1.0)
            nota = piano(librosa.midi_to_hz(m + afinacao), max(0.1, dur - j * 0.015), sr, v, rng)
            a0 = int(ti * sr)
            if a0 >= n or a0 < 0:
                continue
            b = min(n, a0 + len(nota))
            pan = 0.5 + 0.25 * np.sin(m * 1.7)  # cada nota um pouco para um lado
            esq[a0:b] += np.sqrt(1 - pan) * nota[: b - a0]
            dir_[a0:b] += np.sqrt(pan) * nota[: b - a0]
    # Trêmulo estéreo lento (o "vai e vem" do Rhodes de mala).
    t = np.arange(n) / sr
    trem = 0.12 * np.sin(2 * np.pi * 3.2 * t)
    return np.stack([esq * (1 + trem), dir_ * (1 - trem)])


def grade_completa(batidas, duracao):
    """Estende uma grade de andamento fixo para a música toda (o teclado
    toca também nos trechos sem bateria, como a parte de baixo e teclado de
    Peace antes da bateria entrar). Devolve (grade, quantas batidas entraram
    antes da primeira)."""
    p = float(np.median(np.diff(batidas)))
    antes = np.arange(batidas[0], -1e-9, -p)[::-1][:-1]
    depois = np.arange(batidas[-1] + p, duracao, p)
    return np.concatenate([antes, batidas, depois]), len(antes)


def onde_toca(lista_acordes, n, sr, conf_minima=0.6):
    """Curva 0/1: onde o teclado novo toca (acordes com harmonia e confiança)."""
    c = np.zeros(n)
    for ini, fim, _, tipo, conf in lista_acordes:
        if tipo is not None and conf >= conf_minima:
            c[int(ini * sr) : int(fim * sr)] = 1
    return c


def tom_pelos_acordes(lista_acordes, perfil=None, empate=0.95):
    """Tom em que cabem os acordes da música: fração do tempo de notas de
    acorde dentro de cada escala, mais (com `perfil`, a distribuição das
    notas do resto e do baixo) a fração das notas tocadas dentro dela.

    O perfil de Krumhansl sozinho confunde o tom com o quarto grau: em Peace
    (Lá♭ - Sol m - Fá m - Mi♭) ele dizia Lá♭ maior, onde o Sol menor nem
    existe; os acordes dizem Mi♭ maior. Quando dois tons ficam quase
    empatados (Neighbors: Lá maior ou Mi maior), a escala devolvida é só o
    que os dois têm em comum: o teclado deixa de fora a nota que só existe
    em um deles (ré ou ré#), e nenhuma extensão briga com a música."""
    total = sum(fim - ini for ini, fim, _, tipo, _ in lista_acordes if tipo is not None) + 1e-9
    notas = []
    for tonica in range(12):
        for modo, graus in (("maior", (0, 2, 4, 5, 7, 9, 11)), ("menor", (0, 2, 3, 5, 7, 8, 10))):
            escala = {(tonica + g) % 12 for g in graus}
            sensivel = (tonica + 11) % 12 if modo == "menor" else None
            nota = 0.0
            for ini, fim, raiz, tipo, conf in lista_acordes:
                if tipo is None:
                    continue
                dentro = 1 - fora_do_tom(raiz, tipo, escala, sensivel) / len(INTERVALOS[tipo])
                nota += (fim - ini) * dentro / total
                # a tônica como acorde frequente desempata (Mi♭ maior x Dó menor)
                if raiz == tonica and (tipo in ("maj", "maj7") if modo == "maior" else tipo in ("min", "min7")):
                    nota += 0.15 * (fim - ini) / total
            if perfil is not None:
                nota += float(sum(perfil[i] for i in escala) / (perfil.sum() + 1e-9))
            notas.append((nota, tonica, modo, escala, sensivel))
    notas.sort(key=lambda x: -x[0])
    nota, tonica, modo, escala, sensivel = notas[0]
    # O concorrente é o melhor tom com outras notas (Mi maior e Dó# menor
    # são as mesmas notas e não contam).
    outro = next((x for x in notas[1:] if x[3] != escala), None)
    if outro is not None and outro[0] >= empate * nota:
        escala = escala & outro[3]
        sensivel = sensivel if sensivel == outro[4] else None
    return tonica, modo, escala, sensivel


def fora_do_tom(raiz, tipo, escala, sensivel=None):
    """Quantas notas do acorde estão fora da escala. A sensível da escala
    menor (sol# em lá menor) só vale como terça do acorde de dominante (Mi
    maior em lá menor); valendo para qualquer acorde, um Mi menor passaria
    no lugar do Mi maior."""
    n = 0
    for i in INTERVALOS[tipo]:
        t = (raiz + i) % 12
        if t in escala:
            continue
        if sensivel is not None and t == sensivel and i == 4 and (raiz + 4) % 12 == sensivel and tipo in ("maj", "7"):
            continue
        n += 1
    return n


def perfil_de_notas(harmonia, baixo, sr, hop=4096):
    """Quanto cada nota (0-11) soa no resto e no baixo, pesado pelo volume."""
    y = harmonia + baixo
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    e = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0][: c.shape[1]]
    return (c * e[None, : c.shape[1]]).sum(1)


def nota_de_baixo(f, dur, sr, vel=0.8):
    """Baixo elétrico abafado (tipo lo-fi): fundamental com um pouco de 2º e
    3º harmônicos, ataque macio e o som morrendo devagar."""
    n = int((dur + 0.08) * sr)
    t = np.arange(n) / sr
    amp = (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.9)
    amp *= np.clip((dur + 0.08 - t) / 0.08, 0, 1)
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * np.exp(-t / 0.15) + 0.12 * np.sin(6 * np.pi * f * t) * np.exp(-t / 0.08)
    return vel * amp * np.tanh(1.5 * y) / np.tanh(1.5)


def linha_de_baixo(lista_acordes, bumbos, n, sr, afinacao=0.0, conf_minima=0.6, rng=None):
    """Baixo novo, colado no bumbo do loop: em cada bumbo, a raiz do acorde
    que soa ali (entre Mi1 e Ré#2), segurada até o bumbo seguinte, com um
    pequeno respiro. Sem acorde claro, não toca. Devolve (áudio, curva 0/1
    de onde há nota)."""
    rng = rng or np.random.default_rng(0)
    y, notas = np.zeros(n), np.zeros(n)
    bumbos = np.sort(np.asarray(bumbos, dtype=float))
    for k, t0 in enumerate(bumbos):
        t1 = bumbos[k + 1] if k + 1 < len(bumbos) else t0 + 1.0
        a = _acorde_na_batida(lista_acordes, t0, min(t1, t0 + 0.5))
        if a is None or a[3] is None or a[4] < conf_minima:
            continue
        midi = 28 + (a[2] - 4) % 12  # Mi1 (28) a Ré#2 (39)
        dur = max(0.1, min(t1 - t0, 2.5) - 0.06)
        nota = nota_de_baixo(librosa.midi_to_hz(midi + afinacao), dur, sr, rng.uniform(0.85, 1.0))
        i = int(t0 * sr)
        if 0 <= i < n:
            m = min(len(nota), n - i)
            y[i : i + m] += nota[:m]
            notas[i : i + m] = 1
    return y, notas


def riff(outro, sr, lista_acordes, escala, afinacao=0.0, rng=None, densidade_minima=1.6, espaco=0.12, alvos=None,
         canta=None, fracao=0.35, fracao_sob_voz=0.15):
    """O gancho do teclado/sample original tocado no piano elétrico.

    Quando o resto original é um gancho rítmico (5-6 ataques por segundo),
    trocá-lo por acordes de dois toques por compasso apaga a cara da música.
    Aqui cada
    ataque forte do resto original vira uma nota do piano no mesmo instante:
    a nota mais forte do chroma ali, na escala da música, na oitava mais perto
    da anterior. Onde o resto soa como acorde (várias notas com a mesma
    força), toca as notas de cima do acorde da música. Só onde o resto tem
    ataques (>= `densidade_minima` por segundo, em janelas de 4 s): um
    assobio (~3 por segundo) entra; acordes longos (~1 por segundo) não.

    O tempo de cada nota é o do ataque original, não uma grade: puxar as
    notas para a semicolcheia mais próxima as movia 23 ms em média (até
    46 ms) e desfazia o balanço do teclado original, que já está no tempo da
    música. E o ataque é medido fino (passo de 1,5 ms, no começo da subida):
    o detector comum marca o pico, ~18 ms depois.
    Com `alvos` (onde a bateria nova bate, s), a nota que cai a menos de
    35 ms de um golpe vai para ele: copiando cada ataque detectado, as notas
    de uma mesma semicolcheia se espalhavam por até 60 ms.
    Poucas notas: só a fração `fracao` mais forte dos ataques vira nota
    e, enquanto o rapper canta (`canta`, curva 0-1 no tempo da música), só
    a `fracao_sob_voz` mais forte; duas notas ficam a pelo menos `espaco` s
    uma da outra (com metade dos ataques o piano já dá ~2 toques por
    segundo, demais para um teclado de fundo).
    Devolve (áudio estéreo (2, n), curva 0/1 de onde o riff toca)."""
    rng = rng or np.random.default_rng(1)
    n = len(outro)
    hop = 512
    env = librosa.onset.onset_strength(y=outro, sr=sr, hop_length=hop)
    q = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, backtrack=False)
    if len(q) < 8:
        return np.zeros((2, n)), np.zeros(n)
    t_on = librosa.frames_to_time(q, sr=sr, hop_length=hop)
    forca = env[q] / (np.percentile(env[q], 95) + 1e-9)
    # Só a metade mais forte: os ataques fracos do resto são, muitas vezes,
    # vazamento (chimbal, voz) e viravam notas soltas fora do lugar.
    fortes = forca >= np.percentile(forca, 50)
    t_on, forca = t_on[fortes], forca[fortes]
    # (os limiares contam sobre a metade que ficou)
    lim, lim_voz = np.percentile(forca, 100 * (1 - 2 * fracao)), np.percentile(forca, 100 * (1 - 2 * fracao_sob_voz))
    # Onde o resto tem ataques.
    ritmico = np.zeros(n)
    for ini in np.arange(0, n / sr, 4.0):
        k = ((t_on >= ini) & (t_on < ini + 4)).sum()
        if k / 4.0 >= densidade_minima:
            ritmico[int(ini * sr) : int((ini + 4) * sr)] = 1
    # Ataque fino: começo da subida de cada nota.
    hop_f = 64
    env_f = librosa.onset.onset_strength(y=outro, sr=sr, hop_length=hop_f)
    t_f = librosa.frames_to_time(librosa.onset.onset_detect(onset_envelope=env_f, sr=sr, hop_length=hop_f, backtrack=True),
                                 sr=sr, hop_length=hop_f)
    c = librosa.feature.chroma_cqt(y=outro, sr=sr, hop_length=hop)
    esq, dir_ = np.zeros(n), np.zeros(n)
    tocou = np.zeros(n)
    anterior = 69
    ultimo = -1.0
    for k, (t, f) in enumerate(zip(t_on, forca)):
        if not ritmico[min(n - 1, int(t * sr))]:
            continue
        sob_voz = canta is not None and canta[min(len(canta) - 1, int(t * sr))] > 0.5
        if f < (lim_voz if sob_voz else lim):
            continue
        perto = t_f[(t_f >= t - 0.05) & (t_f <= t + 0.01)]
        t0 = float(perto[np.argmin(np.abs(perto - (t - 0.018)))]) if len(perto) else t - 0.018
        t0 = no_golpe(t0, alvos)
        if t0 - ultimo < espaco:
            continue
        ultimo = t0
        prox = t_on[k + 1] if k + 1 < len(t_on) else t0 + 0.5
        # Notas curtas (até 0,4 s): segurando até 1 s, o piano soa quase o
        # tempo todo.
        dur = float(np.clip(prox - t0, 0.08, 0.4))
        a0 = librosa.time_to_frames(t, sr=sr, hop_length=hop)
        v = c[:, a0 : a0 + 4].mean(1)
        if v.max() <= 0:
            continue
        ordem = np.argsort(v)[::-1]
        if v[ordem[1]] > 0.8 * v[ordem[0]]:
            # Soa como acorde: só a nota de cima do acorde da música ali (as
            # três notas, várias vezes por segundo, enchem demais).
            ac = _acorde_na_batida(lista_acordes, t0, t0 + 0.2)
            if ac is None or ac[3] is None:
                continue
            notas = vozes(ac[2], ac[3], None, escala)[-1:]
        else:
            pc = int(ordem[0])
            if escala and pc not in escala:
                pc = min(escala, key=lambda x: min((x - pc) % 12, (pc - x) % 12))
            opcoes = [pc + 12 * o for o in range(4, 7)]
            m = min(opcoes, key=lambda x: abs(x - anterior))
            m = int(np.clip(m, 57, 84))
            anterior = m
            notas = [m]
        vel = float(np.clip(0.45 + 0.4 * f, 0.4, 0.9))
        for i, m in enumerate(notas):
            nota = piano(librosa.midi_to_hz(m + afinacao), dur, sr, vel * (1.0 if len(notas) == 1 else 0.8), rng)
            a = int((t0 + 0.007 * i) * sr)
            if a < 0 or a >= n:
                continue
            b = min(n, a + len(nota))
            pan = 0.5 + 0.2 * np.sin(m * 1.3)
            esq[a:b] += np.sqrt(1 - pan) * nota[: b - a]
            dir_[a:b] += np.sqrt(pan) * nota[: b - a]
            tocou[a:b] = 1
    return np.stack([esq, dir_]), np.maximum(tocou, 0) * ritmico


# --- O gancho como um pianista tocaria -------------------------------------
# Copiado nota a nota (acima), o riff vira notas secas e soltas. Mas em muito
# rap o resto original é um sample de 1-4 compassos repetido a música toda,
# quase igual (correlação 0,7-0,9 com a média). Então: acha o ciclo, tira a média de todas
# as repetições (a voz que vaza e as variações somem), transcreve essa frase
# uma vez e a toca limpa em todo ciclo, com cada acorde soando até o próximo.


def _semitons(y, sr, afinacao=0.0, hop=128):
    """Magnitude por semitom, de C2 a B7 (72 linhas), e o tempo de cada quadro."""
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    bpo = 36
    C = np.abs(librosa.cqt(y, sr=22050, hop_length=hop, fmin=librosa.note_to_hz("C2"), n_bins=6 * bpo,
                           bins_per_octave=bpo, tuning=afinacao))
    # os 3 bins centrados em cada semitom
    S = np.stack([C[max(0, 3 * k - 1) : 3 * k + 2].max(0) for k in range(C.shape[0] // 3)])
    return S, librosa.frames_to_time(np.arange(S.shape[1]), sr=22050, hop_length=hop)


def _semicolcheias(tempos):
    """Cada tempo (s) dividido em 4."""
    t = np.asarray(tempos, dtype=float)
    g = (t[:-1, None] + np.diff(t)[:, None] * np.arange(4) / 4).ravel()
    return np.append(g, t[-1])


def _transcrever(M, pf, limiar=0.22):
    """Notas da frase média `M` (semitons x quadros, `pf` quadros por
    semicolcheia, cíclica): (início, duração, nota MIDI, força), em
    semicolcheias. Os harmônicos de cada nota são descontados das de cima,
    de baixo para cima; uma nota começa onde a energia aparece ou sobe de
    novo (re-ataque) e acaba onde some. A frase é um ciclo: a nota que ataca
    no fim continua no começo."""
    res = np.expm1(M)
    nb = res.shape[0]
    for p in range(nb):
        for h, w in ((12, 0.5), (19, 0.35), (24, 0.25), (28, 0.2), (31, 0.15)):
            if p + h < nb:
                res[p + h] = np.maximum(0, res[p + h] - w * res[p])
    fatias = res.shape[1] // pf
    V = np.stack([res[:, i * pf : (i + 1) * pf].max(1) for i in range(fatias)], 1)
    ref = np.percentile(V, 99.5) + 1e-12
    ativo = V > limiar * ref
    ini = np.stack([res[:, i * pf : i * pf + 3].max(1) for i in range(fatias)], 1)
    fim = np.stack([res[:, np.arange(i * pf - 2, i * pf) % res.shape[1]].mean(1) for i in range(fatias)], 1)
    reataque = ini > 1.6 * fim + 0.05 * ref
    notas = []
    for p in range(nb):
        for i in range(fatias):
            if ativo[p, i] and (not ativo[p, i - 1] or reataque[p, i]):
                d = 1
                while d < fatias and ativo[p, (i + d) % fatias] and not reataque[p, (i + d) % fatias]:
                    d += 1
                notas.append((i, d, 36 + p, float(V[p, [(i + k) % fatias for k in range(d)]].max() / ref)))
    return sorted(notas)


def _arranjar(notas, fatias, min_forca=0.3):
    """A frase transcrita como um pianista a tocaria, em dois níveis:
    'cheio' (acordes e a melodia de cima) e 'esparso' (para baixo da voz: no
    máximo um acorde por tempo, o mais forte). Notas fracas ou de uma
    semicolcheia só (tremidas da transcrição) saem; cada acorde tem até 4
    notas (a de cima e as 3 mais fortes), entre C3 e E6, e soa até o próximo
    acorde (até 2 tempos); a nota de melodia, até o próximo toque. Dois
    toques ficam a pelo menos uma colcheia um do outro.
    Devolve {'cheio': [...], 'esparso': [...]}, cada evento
    (início, duração, [notas], [forças], 'acorde'|'melodia') em semicolcheias."""
    boas = []
    for i, d, m, f in notas:
        if f < min_forca or (d == 1 and f < 0.5):
            continue
        # Abaixo de C3 é o baixo (o baixo novo cuida dele); subido uma oitava,
        # um baixo de sintetizador viraria a "melodia".
        if m < 48:
            continue
        while m > 88:
            m -= 12
        boas.append((i, d, m, f))
    grupos = {}
    for i, d, m, f in boas:
        g = grupos.setdefault(i, {})
        g[m] = max(f, g.get(m, 0))
    # A nota que ataca uma semicolcheia antes ou depois de um toque mais forte
    # entra nele (uma nota de cima que chega 1/16 antes do acorde, separada,
    # vira uma nota solta).
    forca = lambda g: sum(g.values())
    for i in sorted(grupos, key=lambda i: forca(grupos[i])):
        for j in ((i - 1) % fatias, (i + 1) % fatias):
            if i in grupos and j in grupos and forca(grupos[j]) > forca(grupos[i]):
                for m, f in grupos.pop(i).items():
                    grupos[j][m] = max(f, grupos[j].get(m, 0))
    toques = []
    for i, g in grupos.items():
        # Duas notas a um semitom no mesmo toque: fica a mais forte (era a
        # nota de passagem do sample, e juntas soam como erro).
        for m in sorted(g, key=lambda m: g[m]):
            if m in g and any(abs(m - x) == 1 and g[x] >= g[m] for x in g if x != m):
                del g[m]
        ms = sorted(g)
        if len(ms) >= 2:
            topo = ms[-1]
            resto = sorted(ms[:-1], key=lambda x: -g[x])[:3]
            ms = sorted(resto + [topo])
            toques.append((i, ms, [g[x] for x in ms], "acorde"))
        else:
            toques.append((i, ms, [g[ms[0]]], "melodia"))
    soma = lambda t: sum(t[2])
    dist = lambda a, b: min((a - b) % fatias, (b - a) % fatias)

    def escolher(cands):
        feitos = []
        for t in sorted(cands, key=lambda t: (t[3] != "acorde", -soma(t))):
            if all(dist(t[0], u[0]) >= 2 for u in feitos):
                feitos.append(t)
        return sorted(feitos)

    def duracoes(sel):
        ev = []
        ac = sorted(t[0] for t in sel if t[3] == "acorde")
        todos = sorted(t[0] for t in sel)
        for i, ms, fs, tipo in sel:
            prox = [(x - i) % fatias or fatias for x in (ac if tipo == "acorde" else todos) if x != i]
            d = min(prox) if prox else fatias
            d = int(np.clip(d, 2, 8 if tipo == "acorde" else 6))
            ev.append((i, d, ms, fs, tipo))
        return ev

    cheio = escolher(toques)
    # esparso: um acorde por tempo (o mais forte), sem melodia
    por_tempo = {}
    for t in cheio:
        if t[3] == "acorde":
            k = t[0] // 4
            if k not in por_tempo or soma(t) > soma(por_tempo[k]):
                por_tempo[k] = t
    esparso = escolher(list(por_tempo.values()))
    return dict(cheio=duracoes(cheio), esparso=duracoes(esparso))


def _variacao(frase, fatias, papel, pos, sob_voz, virada):
    """A frase de um ciclo com as variações de arranjo de um produtor. A mesma
    frase, igual nos 33 ciclos de 4r Da Squaw, soou "um pouquinho
    repetitiva". O sample continua (é a cara da música), mas:
      introdução   sem variação nenhuma. Ali não há bateria, e o piano fica
                   em primeiro plano: o primeiro ciclo só com a melodia e a
                   virada antes do verso "não ficaram muito bons" (o resto
                   das variações, sim)
      refrão       a frase cheia mesmo com voz, e a nota de cima dobrada uma
                   oitava acima, mais leve (o refrão "abre")
      verso        a versão esparsa; a cada 8 compassos alterna entre os
                   acordes com a nota de cima e sem ela (mais grave). Sem
                   acorde rolado: 30 ms entre as notas punham 11% dos
                   ataques a mais de 25 ms da grade, e um dedilhado de 15 ms
                   já soava "fora do tempo"
      virada       no último ciclo de cada 8 compassos e antes de mudar de
                   seção, o último quarto do ciclo sai: sob a voz fica o
                   acorde anterior soando (respiro); sem voz, as notas de
                   cima uma oitava acima, soltas
    `pos`: o ciclo dentro da seção (0, 1, ...). Devolve os eventos."""
    if papel == "intro":
        return list(frase["esparso" if sob_voz else "cheio"])
    cheio = papel in ("refrao", "instrumental") or not sob_voz
    ev = [list(e) for e in frase["cheio" if cheio else "esparso"]]
    if papel in ("refrao", "instrumental"):
        for e in ev:
            if e[4] == "acorde" and e[2][-1] + 12 <= 88:
                e[2], e[3] = e[2] + [e[2][-1] + 12], e[3] + [0.55 * e[3][-1]]
    elif sob_voz and (pos // 4) % 2 == 1:
        for e in ev:
            if len(e[2]) >= 3:
                e[2], e[3] = e[2][:-1], e[3][:-1]
    if virada:
        corte = fatias * 3 // 4
        fica = [e for e in ev if e[0] < corte]
        sai = [e for e in ev if e[0] >= corte]
        if fica:
            fica[-1][1] = max(fica[-1][1], min(fatias - fica[-1][0], 8))
        if not sob_voz:
            fica += [[i, max(2, d), [ms[-1] + 12], [0.8 * fs[-1]], "melodia"] for i, d, ms, fs, _ in sai if ms[-1] + 12 <= 91]
        ev = fica
    return [tuple(e) for e in ev]


def gancho(outro, sr, tempos, afinacao=0.0, rng=None, alvos=None, canta=None, minimo=0.5, parecido=0.5,
           secoes=None, variar=True, junta=0.6):
    """O sample/teclado que se repete na música, transcrito pela média das
    repetições e tocado no piano elétrico como uma frase (ver acima).

    Em 4r Da Squaw o sample é um só na música toda (semelhança 0,61 a 2
    compassos na música inteira): uma frase para todos os ciclos. Nas
    outras, cada parte tem o seu: em Blinding Lights o verso repete 1 compasso (0,85) e o
    refrão 4 (0,63-0,89); em Neighbors e Count Me Out só algumas partes têm
    ciclo claro. Então, por seção (`secoes`, {ini, fim, papel} no tempo
    original): o ciclo é o de 1, 2 ou 4 compassos com o resto mais parecido
    consigo mesmo um ciclo depois (cosseno >= `minimo`), e as seções com o
    mesmo ciclo e médias parecidas (correlação >= `junta`) formam uma frase
    só, transcrita pela média de todas as repetições. Nas seções sem frase
    própria, cada ciclo que se parece com uma frase a toca. Os
    ciclos ficam numa grade única de compassos a partir de `tempos[0]`, e
    cada um vai para a seção onde está o seu meio.

    tempos: os tempos da música (s, no tempo original), o primeiro abrindo
    um compasso de 4. A frase toca nos ciclos parecidos com a média dela
    (correlação >= `parecido`); onde não toca, fica o teclado de acordes.
    Nos ciclos em que o rapper canta (`canta`, curva 0-1, em mais de 40% do
    ciclo) toca a versão esparsa. Com `variar`, o arranjo muda por seção e
    a cada 8 compassos (_variacao). Devolve (áudio (2, n), curva 0/1 de onde
    toca, info) ou None quando nenhuma seção tem um ciclo assim."""
    rng = rng or np.random.default_rng(3)
    n = len(outro)
    tempos = np.asarray(tempos, dtype=float)
    if len(tempos) < 17:
        return None
    S, tq = _semitons(outro, sr, afinacao)
    L = np.log1p(1000 * S / (S.max() + 1e-12))
    g16 = _semicolcheias(tempos)
    idx = lambda t: np.clip(np.searchsorted(tq, t), 0, len(tq) - 1)
    F = np.stack([L[:, idx(a) : max(idx(a) + 1, idx(b))].max(1) for a, b in zip(g16[:-1], g16[1:])], 1)
    ativo = F.sum(0) > 0.2 * np.percentile(F.sum(0), 90)
    X = F - F.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-9
    nf, pf = F.shape[1], 8
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    if not secoes:
        secoes = [dict(ini=-1.0, fim=float(g16[-1]) + 1.0, papel="verso")]

    def ciclo(i, c):
        """O ciclo de c compassos que começa na semicolcheia i (pf quadros por semicolcheia)."""
        g = np.concatenate([np.linspace(g16[i + j], g16[i + j + 1], pf, endpoint=False) for j in range(16 * c)])
        return L[:, idx(g)]

    def secao_de(i, c):
        meio = (g16[i] + g16[i + 16 * c]) / 2
        for j, s in enumerate(secoes):
            if s["ini"] <= meio < s["fim"]:
                return j
        return 0 if meio < secoes[0]["ini"] else len(secoes) - 1

    def ciclos_da(j, c):
        return [i for i in range(0, nf - 16 * c + 1, 16 * c) if secao_de(i, c) == j]

    def semelhanca(sel, c, minimo_ciclos):
        lag = 16 * c
        if sel.sum() < minimo_ciclos * lag:
            return None
        m = ativo[:-lag] & ativo[lag:] & sel[:-lag] & sel[lag:]
        return float((X[:, :-lag] * X[:, lag:]).sum(0)[m].mean()) if m.sum() >= 16 else None

    def ciclo_de(sims):
        sims = {c: v for c, v in sims.items() if v is not None}
        if not sims or max(sims.values()) < minimo:
            return None
        melhor = max(sims.values())
        return min(c for c, v in sims.items() if v >= 0.9 * melhor), melhor

    # 1) Um sample só na música toda: uma frase para todos os ciclos.
    tudo = np.ones(nf, bool)
    global_ = ciclo_de({c: semelhanca(tudo, c, 3) for c in (1, 2, 4)})
    por_secao = {}
    if global_:
        grupos = [dict(c=global_[0], secs=list(range(len(secoes))))]
        por_secao = {-1: global_}
    else:
        # Senão, o ciclo de cada seção (com pelo menos 2 repetições: há
        # partes com ciclo claro de só 4 compassos).
        for j, s in enumerate(secoes):
            sel = (g16[:-1] >= s["ini"]) & (g16[:-1] < s["fim"])
            r = ciclo_de({c: semelhanca(sel, c, 2) for c in (1, 2, 4)})
            if r:
                por_secao[j] = r
        if not por_secao:
            return None
        # 2) Seções com o mesmo ciclo e a mesma cara viram uma frase só.
        grupos = []
        for j, (c, _) in por_secao.items():
            cs = ciclos_da(j, c)
            if len(cs) < 2:
                continue
            Mj = np.median([ciclo(i, c) for i in cs], 0)
            for g in grupos:
                if g["c"] == c and np.mean(z(Mj) * z(g["M"])) >= junta:
                    g["secs"].append(j)
                    g["M"] = np.median([ciclo(i, c) for jj in g["secs"] for i in ciclos_da(jj, c)], 0)
                    break
            else:
                grupos.append(dict(c=c, secs=[j], M=Mj))
    # 3) A frase de cada grupo, pela média dos ciclos que se parecem com ela.
    tocar = []
    for gi, g in enumerate(grupos):
        c = g["c"]
        cs = [i for j in g["secs"] for i in ciclos_da(j, c)]
        K = np.array([ciclo(i, c) for i in cs])
        energia = K.mean((1, 2))
        com_som = energia > 0.3 * np.percentile(energia, 90)
        M = np.median(K[com_som], 0)
        cor = np.array([np.mean(z(k) * z(M)) for k in K])
        bons = com_som & (cor >= parecido)
        if bons.sum() < 3:
            continue
        M = np.median(K[bons], 0)
        cor = np.array([np.mean(z(k) * z(M)) for k in K])
        bons = com_som & (cor >= parecido)
        frase = _arranjar(_transcrever(M, pf), 16 * c)
        if not frase["cheio"]:
            continue
        # Com as notas de cima abaixo de C4, a "frase" é uma linha de baixo
        # (um baixo de sintetizador que a separação deixou no resto), não
        # teclado.
        if np.median([ms[-1] for _, _, ms, _, _ in frase["cheio"]]) < 60:
            g["baixo"] = True
            continue
        g.update(frase=frase, M=M, piso=0.3 * np.percentile(energia, 90), tocou=0, de=len(cs))
        tocar += [(i, gi) for i, b in zip(cs, bons) if b]
    if not tocar:
        return None
    # 3b) Seções fora das frases (curtas, sem ciclo próprio, ou cuja frase não
    # vingou): cada ciclo que se parece com uma frase toca essa frase. Aqui a
    # exigência é maior (`parecido_fora`): uma frase tocada sobre outra
    # harmonia erra as notas.
    parecido_fora = max(parecido, 0.6)
    feitas = [gi for gi, g in enumerate(grupos) if "frase" in g]
    cobertas = {j for gi in feitas for j in grupos[gi]["secs"]}
    for j in range(len(secoes)):
        if j in cobertas:
            continue
        for i0, gi in ((i, gi) for gi in feitas for i in ciclos_da(j, grupos[gi]["c"])):
            k = ciclo(i0, grupos[gi]["c"])
            if k.mean() > grupos[gi]["piso"] and np.mean(z(k) * z(grupos[gi]["M"])) >= parecido_fora:
                tocar.append((i0, gi))
    tocar.sort()
    # 4) Toca.
    esq, dir_ = np.zeros(n), np.zeros(n)
    tocou = np.zeros(n)
    toques = 0
    variados = 0
    ocupado = 0
    papeis = []
    for i0, gi in tocar:
        c, frase = grupos[gi]["c"], grupos[gi]["frase"]
        fatias = 16 * c
        if i0 < ocupado:  # um ciclo longo de outra frase já toca aqui
            continue
        ocupado = i0 + fatias
        j = secao_de(i0, c)
        papel = secoes[j]["papel"]
        pos = ciclos_da(j, c).index(i0)
        a0, a1 = int(g16[i0] * sr), int(g16[i0 + fatias] * sr)
        sob_voz = canta is not None and np.mean(canta[a0:a1] > 0.5) > 0.4
        if variar:
            prox = i0 + fatias
            virada = pos % 4 == 3 or (prox + fatias <= nf and secao_de(prox, c) != j)
            eventos = _variacao(frase, fatias, papel, pos, sob_voz, virada)
            variados += eventos != list(frase["esparso" if sob_voz else "cheio"])
        else:
            eventos = frase["esparso" if sob_voz else "cheio"]
        papeis.append(f"{papel[:3]}{pos}" + (f"/{gi}" if len(grupos) > 1 else ""))
        grupos[gi]["tocou"] += 1
        for ini, dur, ms, fs, tipo in eventos:
            t0 = no_golpe(float(g16[i0 + ini]), alvos)
            fim = i0 + ini + dur
            d = float(g16[fim] - g16[i0 + ini]) if fim < len(g16) else dur * float(np.median(np.diff(g16)))
            toques += 1
            for k, (m, f) in enumerate(zip(ms, fs)):
                topo = k == len(ms) - 1
                vel = float(np.clip(0.5 + 0.25 * f, 0.45, 0.85)) * (1.05 if topo else 0.9) * rng.uniform(0.92, 1.0)
                nota = piano(librosa.midi_to_hz(m + afinacao), max(0.15, d - 0.03), sr, vel, rng)
                a = int((t0 + 0.005 * k + rng.normal(0, 0.002)) * sr)
                if a < 0 or a >= n:
                    continue
                b = min(n, a + len(nota))
                pan = 0.5 + 0.2 * np.sin(m * 1.7)
                esq[a:b] += np.sqrt(1 - pan) * nota[: b - a]
                dir_[a:b] += np.sqrt(pan) * nota[: b - a]
        tocou[a0:a1] = 1
    t = np.arange(n) / sr
    trem = 0.1 * np.sin(2 * np.pi * 3.2 * t)
    seg = max(1e-9, tocou.sum() / sr)
    nome = lambda ms: [librosa.midi_to_note(m, unicode=False) for m in ms]
    info = dict(toques_por_s=round(toques / seg, 2), cobre=round(float(tocou.mean()), 2), ciclos_variados=int(variados),
                ciclo_por_secao={(f"{secoes[j]['papel']} {secoes[j]['ini']:.0f}s" if j >= 0 else "música toda"): f"{c} comp. ({v:.2f})" for j, (c, v) in por_secao.items()},
                frases=[dict(compassos=g["c"], secoes=[f"{secoes[j]['papel']} {secoes[j]['ini']:.0f}s" for j in g["secs"]],
                             tocados=g.get("tocou", 0), de=g.get("de", 0), linha_de_baixo=g.get("baixo", False),
                             notas=[(i, d, nome(ms), tipo) for i, d, ms, _, tipo in g["frase"]["cheio"]] if "frase" in g else None)
                        for g in grupos],
                papeis=" ".join(papeis))
    return np.stack([esq * (1 + trem), dir_ * (1 - trem)]), tocou, info


# --- Teclado em progressão, como nas lo-fis de rap do YouTube ---------------
# Em remixes lo-fi de rap populares o produtor não transcreve o sample: toca
# uma progressão simples nos acordes da música, acordes de 3 notas no meio do
# teclado, cada um soando 2-4 tempos, o teclado soando 75-100% do tempo, e a
# frase se repetindo a cada 4-8 compassos.


def _tres_notas(raiz, tipo, escala, anterior=None, com_fundamental=False):
    """Acorde de 3 notas sem a fundamental (o baixo novo toca a fundamental):
    terça, sétima (ou sexta) e nona quando cabem no tom; senão fundamental,
    terça e sétima, ou a tríade. Cada nota entre F3 e E5, perto das do acorde
    anterior (condução de vozes)."""
    iv = notas_lofi(raiz, tipo, escala)
    terca = [i for i in iv if i in (3, 4)]
    setima = [i for i in iv if i in (9, 10, 11)]
    if com_fundamental:
        esc = [0, terca[0], setima[0]] if terca and setima else ([0, terca[0], 7] if terca else [0, 7, 12])
    elif terca and setima and 14 in iv:
        esc = [terca[0], setima[0], 14]
    elif terca and setima:
        esc = [0, terca[0], setima[0]]
    elif terca:
        esc = [0, terca[0], 7]
    else:
        esc = [0, 7, 12]
    pcs = [(raiz + i) % 12 for i in esc]
    opcoes = [[p + 12 * o for o in range(3, 7) if 53 <= p + 12 * o <= 76] for p in pcs]
    alvo = sorted(anterior) if anterior else [60, 64, 67]
    melhor, custo = None, 1e9
    for a in opcoes[0]:
        for b in opcoes[1]:
            for c in opcoes[2]:
                v = sorted((a, b, c))
                if v[-1] - v[0] > 14 or len(set(v)) < 3:
                    continue
                k = sum(abs(x - y) for x, y in zip(v, alvo))
                if k < custo:
                    melhor, custo = v, k
    return melhor or sorted(o[0] for o in opcoes)


def progressao(lista_acordes, tempos, n, sr, escala, afinacao=0.0, rng=None, alvos=None, canta=None,
               secoes=None, variar=True, conf_minima=0.5, com_fundamental=True, usar_ciclo=False, ciclo_minimo=0.8):
    """Teclado novo em progressão (ver o estudo acima), no ritmo harmônico da
    própria música.

    A primeira versão tocava um acorde por compasso e soou como "nota
    errada", mesmo com o Rhodes gravado ("o problema são as notas mesmo").
    Medido em No Role Modelz: o baixo troca de acorde num ritmo de 5 + 3
    tempos (A#m 5, Fm 3, G# 5, C# 3), e um acorde por compasso sempre errava
    uma das duas trocas por um tempo: o piano ficava no acorde velho em cima
    do novo da música. Então, tempo a tempo: o acorde de cada tempo vem de
    `lista_acordes`; o ciclo é o de 4, 8, 16 ou 32 tempos que mais se repete,
    e cada posição dele toca o acorde mais comum ali (os erros de detecção
    somem); um acorde de um tempo só vira o vizinho. O piano troca de acorde
    exatamente onde a música troca.

    tempos: os tempos da música (s, tempo original); com a grade em
    meio-tempo (0,9 s ou mais por tempo), cada tempo vira dois.
    Em cada acorde: o toque no começo e, se ele dura 3 tempos ou mais, de
    novo mais leve no "e" do 3º tempo dele. Sem voz (e com bateria), a cada
    dois acordes a nota de cima sobe um grau no 3º tempo e, a cada quatro, o
    próximo acorde é antecipado em uma colcheia. No refrão, a nota de cima
    dobrada uma oitava acima no primeiro toque. Sem bateria (introdução), o
    acorde soa inteiro, sem enfeites. Acorde com a fundamental (mais perto do
    que a música toca: cosseno do chroma 0,65 contra 0,53 sem ela).
    Devolve (áudio (2, n), curva 0/1 de onde toca, info) ou None."""
    rng = rng or np.random.default_rng(5)
    tempos = np.asarray(tempos, dtype=float)
    if len(tempos) < 9 or not lista_acordes:
        return None
    if np.median(np.diff(tempos)) >= 0.9:
        tempos = np.sort(np.concatenate([tempos, tempos[:-1] + np.diff(tempos) / 2]))
    nt = len(tempos) - 1
    chave = lambda r, t: (r, t in ("min", "min7"))
    por_tempo = []
    for a, b in zip(tempos[:-1], tempos[1:]):
        ac = _acorde_na_batida(lista_acordes, a, b)
        ok = ac is not None and ac[3] is not None and ac[4] >= conf_minima
        por_tempo.append((ac[2], ac[3]) if ok else None)
    if sum(x is not None for x in por_tempo) < 16:
        return None

    def acerto(c):
        p = [(x, y) for x, y in zip(por_tempo[:-c], por_tempo[c:]) if x and y]
        return np.mean([chave(*x) == chave(*y) for x, y in p]) if len(p) >= 8 else 0.0

    ac = {c: acerto(c) for c in (4, 8, 16, 32)}
    melhor = max(ac.values())
    ciclo = min(c for c, v in ac.items() if v >= melhor - 0.05)
    prog = []
    for p in range(ciclo):
        aqui = [x for x in por_tempo[p::ciclo] if x]
        if not aqui:
            prog.append(None)
            continue
        k, _ = Counter(chave(*x) for x in aqui).most_common(1)[0]
        prog.append((k[0], Counter(t for r, t in aqui if chave(r, t) == k).most_common(1)[0][0]))
    for p in range(ciclo):  # nada de acorde de um tempo só
        if prog[p] and prog[p] != prog[p - 1] and prog[p] != prog[(p + 1) % ciclo] and prog[p - 1]:
            prog[p] = prog[p - 1]
    # Onde a música tem harmonia (um tempo sem acorde entre dois com acorde também toca).
    tem = [por_tempo[k] is not None or (0 < k < nt - 1 and por_tempo[k - 1] is not None and por_tempo[k + 1] is not None)
           for k in range(nt)]
    # Padrão: os acordes da própria música tempo a tempo, sem os de um tempo
    # só (viram o anterior) e com os buracos de um tempo tapados. Com
    # `usar_ciclo` (True, ou "auto" com acerto >= `ciclo_minimo`), a
    # progressão do ciclo. Tempo a tempo erra menos: a nota mais forte do
    # baixo cai dentro do acorde do piano em 80-98% do tempo, contra 41-84%
    # com o ciclo (as partes costumam ter progressões diferentes).
    modo = "ciclo" if (usar_ciclo is True or (usar_ciclo == "auto" and melhor >= ciclo_minimo)) else "tempo a tempo"
    if modo == "ciclo":
        tocar = [prog[k % ciclo] if tem[k] else None for k in range(nt)]
    else:
        tocar = list(por_tempo)
        for k in range(1, nt - 1):
            if tocar[k] is None and tem[k]:
                tocar[k] = tocar[k - 1]
        mudou = True
        while mudou:
            mudou = False
            for k in range(1, nt):
                if tocar[k] and tocar[k - 1] and chave(*tocar[k]) != chave(*tocar[k - 1]) and (k + 1 >= nt or tocar[k + 1] is None or chave(*tocar[k + 1]) != chave(*tocar[k])):
                    tocar[k] = tocar[k - 1]
                    mudou = True
        tocar = [(x[0], x[1]) if x else None for x in tocar]
    # Acordes: tempos seguidos com o mesmo acorde (no máximo 8 tempos sem tocar de novo).
    segs = []
    k = 0
    while k < nt:
        if tocar[k] is None:
            k += 1
            continue
        k1 = k + 1
        while k1 < nt and tocar[k1] == tocar[k] and k1 - k < 8:
            k1 += 1
        segs.append((k, k1, tocar[k]))
        k = k1
    if not segs:
        return None
    papel_de = lambda t: next((s["papel"] for s in (secoes or []) if s["ini"] <= t < s["fim"]), "verso")
    esq, dir_ = np.zeros(n), np.zeros(n)
    tocou = np.zeros(n)
    toques = 0
    escala_ord = sorted(escala) if escala else list(range(12))

    def em(k0, x):
        """Instante do tempo k0 + x (x em tempos, fracionário)."""
        i = min(nt - 1, k0 + int(x))
        return tempos[i] + (x - int(x)) * (tempos[i + 1] - tempos[i])

    def soar(t, dur, notas, vel):
        nonlocal toques
        toques += 1
        t = no_golpe(t, alvos)
        for j, m in enumerate(sorted(notas)):
            topo = j == len(notas) - 1
            v = vel * (1.05 if topo else 0.9) * rng.uniform(0.92, 1.0)
            nota = piano(librosa.midi_to_hz(m + afinacao), max(0.12, dur - 0.03), sr, v, rng)
            a0 = int((t + 0.005 * j + rng.normal(0, 0.002)) * sr)
            if a0 < 0 or a0 >= n:
                continue
            b0 = min(n, a0 + len(nota))
            pan = 0.5 + 0.2 * np.sin(m * 1.7)
            esq[a0:b0] += np.sqrt(1 - pan) * nota[: b0 - a0]
            dir_[a0:b0] += np.sqrt(pan) * nota[: b0 - a0]

    def grau_acima(m):
        for d in range(1, 4):
            if (m + d) % 12 in escala_ord:
                return m + d
        return m + 2

    anterior = None
    antecipado = False
    for i, (k0, k1, a) in enumerate(segs):
        L = k1 - k0
        ini, fim = tempos[k0], tempos[k1]
        notas = _tres_notas(a[0], a[1], escala, anterior, com_fundamental)
        anterior = notas
        papel = papel_de(ini)
        a0, a1 = int(ini * sr), int(fim * sr)
        sob_voz = canta is not None and np.mean(canta[a0:a1] > 0.5) > 0.4
        simples = not variar or papel == "intro"
        prox = segs[i + 1] if i + 1 < len(segs) and segs[i + 1][0] == k1 else None
        if not antecipado:
            dobra = [notas[-1] + 12] if variar and papel in ("refrao", "instrumental") and notas[-1] + 12 <= 88 else []
            fim1 = fim if simples or L < 3 else em(k0, 2.5)
            soar(ini, fim1 - ini, notas + dobra, 0.62)
        antecipado = False
        if not simples:
            if L >= 3:
                soar(em(k0, 2.5), fim - em(k0, 2.5), notas, 0.42 if sob_voz else 0.45)
            if not sob_voz and i % 2 == 1 and L >= 4:
                soar(em(k0, 2.0), em(k0, 3.0) - em(k0, 2.0), [grau_acima(notas[-1])], 0.5)
            if not sob_voz and i % 4 == 3 and prox is not None and L >= 2:
                nxt = _tres_notas(prox[2][0], prox[2][1], escala, notas, com_fundamental)
                t_ant = em(k0, L - 0.5)
                soar(t_ant, tempos[prox[1]] - t_ant if prox[1] - prox[0] < 3 else em(prox[0], 2.5) - t_ant, nxt, 0.55)
                antecipado = True
        tocou[a0:a1] = 1
    if not toques:
        return None
    t = np.arange(n) / sr
    trem = 0.1 * np.sin(2 * np.pi * 3.2 * t)
    seg = max(1e-9, tocou.sum() / sr)
    ritmo, atual, conta = [], None, 0
    for x in prog:
        if x != atual and atual is not None:
            ritmo.append(f"{nome(*atual)}({conta})")
            conta = 0
        atual, conta = x, conta + 1
    if atual is not None:
        ritmo.append(f"{nome(*atual)}({conta})")
    info = dict(estilo="progressao", modo=modo, ciclo_tempos=ciclo, acerto=round(melhor, 2), progressao=" ".join(ritmo),
                com_fundamental=com_fundamental, cobre=round(float(tocou.mean()), 2), toques_por_s=round(toques / seg, 2))
    return np.stack([esq * (1 + trem), dir_ * (1 - trem)]), tocou, info
