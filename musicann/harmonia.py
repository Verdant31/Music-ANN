"""Teclado novo: os acordes da música tocados num piano elétrico lo-fi.

As lo-fi mais ouvidas do YouTube (Yuko Records, jdutra, Smooth Sounds...)
são remixes: o produtor guarda a voz e refaz a base, com acordes "jazzy"
(com sétima e nona) num piano elétrico tipo Rhodes. Manter o instrumental
original, só filtrado, deixa a lo-fi parecida demais com a música.

  acordes()   acha os acordes da música no baixo e no resto (chroma contra
              moldes de acordes, suavizado com Viterbi), com uma confiança
  vozes()     escolhe as notas de cada acorde (com sétima e nona, perto
              das notas do acorde anterior, como um tecladista tocaria)
  rhodes()    sintetiza uma nota de piano elétrico (FM, como o DX7)
  teclado()   toca os acordes na grade de compassos, com o ritmo e as
              pequenas imperfeições de um tecladista
"""
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


def acordes(harmonia, baixo, sr, hop=4096, troca=0.995, escala=None):
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
    raiz_baixo = cb / (cb.max(0, keepdims=True) + 1e-9)
    bonus = np.array([raiz_baixo[r] for r, _ in rotulos]) * 0.25
    # Tríade simples tem uma leve preferência (sétima só quando ela está lá).
    simples = np.array([0.03 if t in ("maj", "min") else 0 for _, t in rotulos])[:, None]
    fora = np.zeros((len(rotulos), 1))
    if escala:
        fora[:, 0] = [-0.06 * sum((r + i) % 12 not in escala for i in TIPOS[t][0]) for r, t in rotulos]
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
        if calado[i : j + 1].mean() > 0.5:
            tipo = None
        saida.append((float(tempos[i]), float(tempos[j + 1]), raiz, tipo, conf))
        i = j + 1
    return saida


def nome(raiz, tipo):
    return "N" if tipo is None else NOMES[raiz] + {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7"}[tipo]


def notas_lofi(raiz, tipo, escala=None):
    """Intervalos do acorde de lo-fi: o acorde detectado com sétima e nona,
    mas só as que pertencem ao tom (uma nona fora do tom, como o lá natural
    num Sol menor dentro de Mi bemol maior, briga com a melodia)."""
    no_tom = lambda i: escala is None or (raiz + i) % 12 in escala
    terca = 3 if tipo in ("min", "min7") else 4
    iv = [0, terca]
    if tipo != "7":
        iv.append(7)
    if tipo == "7":
        setima = 10
    elif tipo == "maj7":
        setima = 11
    elif tipo == "min7":
        setima = 10
    else:  # tríade: a sétima que estiver no tom
        setima = next((x for x in ((11, 10) if tipo == "maj" else (10, 11)) if no_tom(x)), None)
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
    """Uma nota de piano elétrico: FM com o modulador na mesma frequência
    (o corpo "redondo") e um segundo modulador agudo que só aparece no
    ataque (o "plim" da lâmina). A nota soa `dur` s e solta em 0,25 s."""
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


def _acorde_na_batida(lista, t0, t1):
    """O acorde que mais soa entre t0 e t1 (ou None)."""
    melhor, tempo = None, 0.0
    for a in lista:
        sob = min(a[1], t1) - max(a[0], t0)
        if sob > tempo:
            melhor, tempo = a, sob
    return melhor


def teclado(lista_acordes, batidas, n, sr, afinacao=0.0, rng=None, conf_minima=0.6, escala=None):
    """Faixa estéreo (2, n) com os acordes tocados no piano elétrico.

    batidas: os tempos do groove (s), o primeiro abrindo um compasso de 4.
    Em cada compasso: acorde cheio no tempo 1 e um toque mais leve no
    contratempo do 2 (o padrão mais comum de teclado em lo-fi); se o acorde
    da música muda em outro tempo, toca a troca ali. Dedilhado de baixo para
    cima em ~15 ms, com o tempo e a força um pouco irregulares. Acordes com
    confiança baixa ou sem harmonia ficam de fora (melhor faltar teclado que
    tocar a nota errada por cima da voz). afinacao: desvio da música em
    relação ao lá 440 (fração de semitom, librosa.estimate_tuning)."""
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
            ti = ini + j * 0.015 + rng.normal(0, 0.006)
            v = vel * rng.uniform(0.85, 1.0) * (0.8 if j == 0 else 1.0)
            nota = rhodes(librosa.midi_to_hz(m + afinacao), max(0.1, dur - j * 0.015), sr, v, rng)
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


def tom_pelos_acordes(lista_acordes, perfil=None, empate=0.97):
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
        for modo, graus in (("maior", (0, 2, 4, 5, 7, 9, 11)), ("menor", (0, 2, 3, 5, 7, 8, 10, 11))):
            escala = {(tonica + g) % 12 for g in graus}
            nota = 0.0
            for ini, fim, raiz, tipo, conf in lista_acordes:
                if tipo is None:
                    continue
                tons = [(raiz + i) % 12 for i in TIPOS[tipo][0]]
                nota += (fim - ini) * np.mean([t in escala for t in tons]) / total
                # a tônica como acorde frequente desempata (Mi♭ maior x Dó menor)
                if raiz == tonica and (tipo in ("maj", "maj7") if modo == "maior" else tipo in ("min", "min7")):
                    nota += 0.15 * (fim - ini) / total
            if perfil is not None:
                nota += float(sum(perfil[i] for i in escala) / (perfil.sum() + 1e-9))
            notas.append((nota, tonica, modo, escala))
    notas.sort(key=lambda x: -x[0])
    nota, tonica, modo, escala = notas[0]
    if notas[1][0] >= empate * nota:
        escala = escala & notas[1][3]
    return tonica, modo, escala


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
