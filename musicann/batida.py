"""Gera uma base lo-fi nova seguindo a harmonia da música, com o MusicGen-melody.

A batida dos lo-fi de verdade é inventada pelo produtor, então nenhuma rede
consegue "prever" a batida certa a partir da original. O que dá para fazer é
gerar uma batida plausível: o MusicGen-melody (Meta) recebe um texto
descrevendo o estilo e o chroma (as notas) do instrumental original, e gera
uma base nova que segue os mesmos acordes.

Gera em pedaços de até 30 s (limite do modelo) e emenda com crossfade.
"""
import numpy as np
import librosa
import torch

MODELO = "facebook/musicgen-melody"
HOP = 512


def andamento(y, sr):
    t, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    return float(np.atleast_1d(t)[0])


def sincronizar(gerado, referencia, sr):
    """Encaixa a base gerada no tempo da música.

    O MusicGen não trava o andamento: a base pode sair alguns bpm mais
    rápida ou lenta, e começar fora da cabeça do tempo. Aqui (1) estico ou
    encolho a base para o andamento da referência, escolhendo entre x1, x2 e
    x0,5 (o detector às vezes confunde o dobro com a metade), e (2) desloco a
    base até as batidas dela coincidirem com as da referência
    (correlação das curvas de ataque, dentro de um tempo).
    """
    t_ref, t_ger = andamento(referencia, sr), andamento(gerado, sr)
    if t_ref > 0 and t_ger > 0:
        r = min((t_ref / t_ger * k for k in (1, 2, 0.5)), key=lambda v: abs(np.log(v)))
        if 0.75 < r < 1.33 and abs(r - 1) > 0.01:
            gerado = librosa.effects.time_stretch(gerado, rate=r)
    o_ref = librosa.onset.onset_strength(y=referencia, sr=sr, hop_length=HOP)
    o_ger = librosa.onset.onset_strength(y=gerado, sr=sr, hop_length=HOP)
    n = min(len(o_ref), len(o_ger))
    o_ref, o_ger = o_ref[:n] - o_ref[:n].mean(), o_ger[:n] - o_ger[:n].mean()
    periodo = int(60 / max(t_ref, 40) * sr / HOP) if t_ref > 0 else 40
    lags = range(-periodo, periodo + 1)
    corr = [np.dot(o_ref[max(0, l):n + min(0, l)], o_ger[max(0, -l):n - max(0, l)]) for l in lags]
    atraso = list(lags)[int(np.argmax(corr))] * HOP
    if atraso > 0:
        gerado = np.concatenate([np.zeros(atraso, gerado.dtype), gerado])
    elif atraso < 0:
        gerado = gerado[-atraso:]
    m = len(referencia)
    return np.pad(gerado, (0, max(0, m - len(gerado))))[:m]
PROMPT = ("lofi hip hop beat, mellow dusty drums, warm rhodes piano, soft bass, "
          "vinyl crackle, chill, {bpm} bpm")


class GeradorDeBatida:
    def __init__(self, dev="cuda"):
        from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

        self.proc = AutoProcessor.from_pretrained(MODELO)
        self.modelo = MusicgenMelodyForConditionalGeneration.from_pretrained(
            MODELO, torch_dtype=torch.float16).to(dev).eval()
        self.sr = self.modelo.config.audio_encoder.sampling_rate
        self.dev = dev
        self.sincronizar = True

    @torch.no_grad()
    def gerar(self, instrumental, sr, bpm, pedaco=30.0, emenda=2.0, semente=0):
        """instrumental: áudio mono (a harmonia que a base deve seguir)."""
        torch.manual_seed(semente)
        y = librosa.resample(instrumental, orig_sr=sr, target_sr=self.sr)
        passo = int((pedaco - emenda) * self.sr)
        tam = int(pedaco * self.sr)
        n_emenda = int(emenda * self.sr)
        saida = np.zeros(len(y) + tam, dtype=np.float32)
        rampa = np.linspace(0, 1, n_emenda, dtype=np.float32)
        for ini in range(0, len(y), passo):
            trecho = y[ini : ini + tam]
            if ini > 0 and len(trecho) < n_emenda + self.sr:
                break  # sobra curta demais: já foi coberta pela emenda anterior
            entrada = self.proc(text=[PROMPT.format(bpm=int(round(bpm)))], audio=trecho,
                                sampling_rate=self.sr, padding=True, return_tensors="pt").to(self.dev)
            entrada = {k: (v.half() if v.is_floating_point() else v) for k, v in entrada.items()}
            n_tokens = int(len(trecho) / self.sr * self.modelo.config.audio_encoder.frame_rate)
            gerado = self.modelo.generate(**entrada, do_sample=True, guidance_scale=3.0,
                                          max_new_tokens=n_tokens)[0, 0].float().cpu().numpy()
            gerado = sincronizar(gerado, trecho, self.sr) if self.sincronizar else gerado[: len(trecho)]
            if ini > 0:
                m = min(n_emenda, len(gerado))
                gerado[:m] *= rampa[:m]
                saida[ini : ini + m] *= rampa[::-1][:m]
            saida[ini : ini + len(gerado)] += gerado
        saida = saida[: len(y)]
        return librosa.resample(saida, orig_sr=self.sr, target_sr=sr)
