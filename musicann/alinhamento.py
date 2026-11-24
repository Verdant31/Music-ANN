"""Alinha a versão lo-fi com a original.

A lo-fi costuma ser mais lenta, cortar trechos e às vezes estar em outro tom.
Para cada bloco de ~3 s da lo-fi procuro, com DTW de subsequência sobre o
chroma (as notas que estão soando), o trecho da original que mais se parece.
Isso dá um mapa quadro da lo-fi -> quadro da original, com uma nota de
confiança por bloco.
"""
import numpy as np
import librosa

from .audio import SR, HOP

BLOCO = 128  # quadros (~3 s)
PASSOS = np.array([[1, 1], [1, 2], [2, 1]])  # inclinação entre 0,5x e 2x
# O passo (2, 1) cobre dois quadros da lo-fi pagando uma célula só; sem peso 2
# ele sai mais barato e o caminho fica artificialmente "lento".
PESOS = np.array([1.0, 1.0, 2.0])


def chroma(y):
    # tuning=None: o librosa estima a afinação, o que absorve desvios de
    # menos de meio tom (comuns em versões "slowed").
    c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, tuning=None)
    # Quadros em silêncio dão chroma zerado, e a distância cosseno vira NaN.
    return c + 1e-6


def deslocamento_de_tom(c_original, c_lofi, rms_lofi, maximo=4):
    """Quantos semitons (inteiros) a lo-fi está acima da original.

    Comparar só o perfil médio de notas confunde o tom com a quarta ou a
    quinta dele. Aqui testo cada deslocamento de -4 a +4 alinhando de fato
    um trecho do meio da música e fico com o de menor custo. Versões
    "slowed" feitas como disco mais lento descem até ~3-4 semitons.
    """
    meio = c_lofi.shape[1] // 2
    ini, fim = max(0, meio - 4 * BLOCO), min(c_lofi.shape[1], meio + 4 * BLOCO)
    custos = {}
    for k in range(-maximo, maximo + 1):
        _, custo, _ = alinhar(np.roll(c_original, k, axis=0), c_lofi[:, ini:fim], rms_lofi[ini:fim])
        ok = np.isfinite(custo)
        custos[k] = np.median(custo[ok]) if ok.any() else np.inf
    return min(custos, key=custos.get)


def feicao_harmonica(m_db):
    """Espectrograma (bandas 10-90) só da parte harmônica, um vetor por quadro.

    A separação harmônico/percussivo (HPSS) tira a bateria, que nos lo-fi
    flips é nova e atrapalha a comparação. O que sobra (voz, acordes) é igual
    nas duas versões e tem resolução de quadro, ao contrário do chroma.
    """
    h, _ = librosa.decompose.hpss(librosa.db_to_power(m_db), margin=2.0)
    f = librosa.power_to_db(h + 1e-10)[10:90]
    f = f - f.mean(axis=0, keepdims=True)
    f = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-6)
    # Quadros "planos" (silêncio) viram vetor zero e a distância cosseno vira NaN.
    return f + 1e-4


def refinar(f_original, f_lofi, mapa, custo, bloco=BLOCO, folga=32):
    """Refina o mapa bloco a bloco com DTW sobre a feição harmônica.

    Procura de novo cada bloco num trecho da original em volta da posição
    achada pelo chroma (com `folga` quadros, ~0,7 s, para cada lado). Devolve
    (mapa, custo_fino), com o custo cosseno médio quadro a quadro do caminho
    escolhido, que serve para filtrar os trechos que continuam ruins.
    """
    T = f_lofi.shape[1]
    mapa = mapa.copy()
    custo_fino = np.full(T, np.inf, dtype=np.float32)
    To = f_original.shape[1]
    for ini in range(0, T, bloco):
        fim = min(ini + bloco, T)
        if not np.isfinite(custo[ini]) or fim - ini < 16:
            continue
        j0 = max(0, int(mapa[ini:fim].min()) - folga)
        j1 = min(To, int(mapa[ini:fim].max()) + folga)
        X, Y = f_lofi[:, ini:fim], f_original[:, j0:j1]
        antes = 1 - np.sum(X * f_original[:, np.clip(mapa[ini:fim], 0, To - 1)], axis=0).mean()
        _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine", subseq=True,
                                     step_sizes_sigma=PASSOS, weights_mul=PESOS)
        wp = wp[::-1]
        i, j = wp[:, 0], wp[:, 1] + j0
        _, idx = np.unique(i, return_index=True)
        novo = np.round(np.interp(np.arange(fim - ini), i[idx], j[idx])).astype(np.int64)
        depois = 1 - np.sum(X * f_original[:, novo], axis=0).mean()
        if depois < antes:
            mapa[ini:fim] = novo
        custo_fino[ini:fim] = min(antes, depois)
    return mapa, custo_fino


def alinhar(c_original, c_lofi, rms_lofi, bloco=BLOCO):
    """Devolve (mapa, custo, inclinacao) com um valor por quadro da lo-fi.

    mapa[i] = quadro da original que corresponde ao quadro i da lo-fi
    custo[i] = custo médio do caminho DTW do bloco (menor é melhor)
    inclinacao[i] = velocidade da original em relação à lo-fi no bloco
    Blocos quase em silêncio ficam com custo infinito.
    """
    T = c_lofi.shape[1]
    mapa = np.zeros(T, dtype=np.int64)
    custo = np.full(T, np.inf, dtype=np.float32)
    inclinacao = np.zeros(T, dtype=np.float32)
    limiar_silencio = 0.1 * np.median(rms_lofi)

    for ini in range(0, T - bloco // 2, bloco):
        fim = min(ini + bloco, T)
        if np.median(rms_lofi[ini:fim]) < limiar_silencio:
            continue
        D, wp = librosa.sequence.dtw(
            X=c_lofi[:, ini:fim], Y=c_original, metric="cosine",
            subseq=True, step_sizes_sigma=PASSOS, weights_mul=PESOS,
        )
        wp = wp[::-1]
        i, j = wp[:, 0], wp[:, 1]
        # Um ponto por quadro da lo-fi, interpolando os saltos dos passos 2x.
        _, idx = np.unique(i, return_index=True)
        j_por_i = np.interp(np.arange(fim - ini), i[idx], j[idx])
        mapa[ini:fim] = np.round(j_por_i).astype(np.int64)
        custo[ini:fim] = D[wp[-1, 0], wp[-1, 1]] / len(wp)
        inclinacao[ini:fim] = (j[-1] - j[0]) / max(i[-1] - i[0], 1)
    return mapa, custo, inclinacao
