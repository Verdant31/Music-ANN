"""Gera trechos de 30 s para comparar, de ouvido, as abordagens numa música
de teste (que nenhum modelo viu no treino).

A voz fica sempre limpa (só desacelerada, como nos lo-fi de verdade); o que
muda entre as versões é a base. Para cada música grava em
exemplos/finais/<música>/:
  0_original.wav                  trecho da música original
  1_lofi_real.wav                 o mesmo trecho na lo-fi do YouTube
  2_voz_limpa_inst_unet.wav       instrumental original com timbre lo-fi (U-Net)
  3_voz_limpa_musicgen.wav        batida lo-fi nova do MusicGen, sincronizada
  4_voz_limpa_musicgen_inst.wav   batida nova + um pouco do instrumental lo-fi

    python exemplos_finais.py Kendrick_Money_Trees Djavan_Oceano --inicio 60
"""
import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from musicann import audio
from musicann.dados import carregar_pares
from gerar_lofi import carregar_modelo, transformar
from musicann.batida import andamento


def tratar(y, modelo, dev):
    db = audio.mel_db(y)
    return audio.aplicar_diferenca(y, db, transformar(modelo, db, dev))


def nivel(y, alvo_db=-20.0):
    return y * 10 ** (alvo_db / 20) / (np.sqrt(np.mean(y**2)) + 1e-12)


def gravar(caminho, y):
    sf.write(caminho, (0.95 * y / (np.abs(y).max() + 1e-8)).astype(np.float32), audio.SR)
    print("gravado:", caminho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("musicas", nargs="+")
    ap.add_argument("--inicio", type=float, default=50.0, help="segundos, na linha do tempo da lo-fi")
    ap.add_argument("--duracao", type=float, default=30.0)
    ap.add_argument("--inst", default="modelos/n_inst/unet.pt")
    ap.add_argument("--sem-musicgen", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    pares = carregar_pares("dados/voz")
    m_inst = carregar_modelo(args.inst, dev)
    gerador = None
    if not args.sem_musicgen:
        from musicann.batida import GeradorDeBatida
        gerador = GeradorDeBatida(dev)

    for nome in args.musicas:
        par = pares[nome]
        ok = np.isfinite(par["custo"])
        veloc = float(np.median(par["inclinacao"][ok]))  # quanto a lo-fi real é mais lenta
        veloc = min(veloc, 1.0)  # nunca acelerar
        q = lambda s: int(s * audio.SR / audio.HOP)
        # O corte é pelo tempo da original; na lo-fi, pego o trecho mais longo
        # cujos quadros apontam para dentro dele.
        o_a, o_b = q(args.inicio), q(args.inicio + args.duracao)
        o_ini, o_fim = o_a * audio.HOP, o_b * audio.HOP
        dentro = ok & (par["mapa"] >= o_a) & (par["mapa"] < o_b)
        idx = np.flatnonzero(dentro)
        cortes = np.split(idx, np.flatnonzero(np.diff(idx) > q(2)) + 1) if len(idx) else [np.array([0])]
        trecho = max(cortes, key=len)
        a, b = int(trecho[0]), int(trecho[-1])
        print(f"{nome}: original {args.inicio:.0f}s-{args.inicio + args.duracao:.0f}s = lo-fi "
              f"{a * audio.HOP / audio.SR:.0f}s-{b * audio.HOP / audio.SR:.0f}s, velocidade {veloc:.2f}, "
              f"tom {float(par['tom']):+.1f}")
        pasta = os.path.join("exemplos", "finais", nome)
        os.makedirs(pasta, exist_ok=True)

        carregar = lambda lado, parte: audio.carregar(os.path.join("Musics", "Separado", lado, f"{nome}_{parte}.wav"))
        voz, inst = carregar("Normal", "voz")[o_ini:o_fim], carregar("Normal", "inst")[o_ini:o_fim]
        n = min(len(voz), len(inst))
        voz, inst = voz[:n], inst[:n]
        mix = voz + inst
        # Lo-fi real: a mesma duração (já desacelerada) a partir do ponto correspondente.
        n_lofi = int(args.duracao / veloc * audio.SR)
        lofi = audio.carregar(os.path.join("Musics", "Lofi", nome + ".mp3"))[a * audio.HOP : a * audio.HOP + n_lofi]
        tom = float(par["tom"])

        def lento(y):
            """Aplica a mesma desaceleração (e mudança de tom) da lo-fi real.
            Se tom e velocidade batem com um disco tocado mais devagar, faço
            isso de uma vez (reamostragem), como os "slowed" do YouTube."""
            if tom < -0.5 and abs(12 * np.log2(veloc) - tom) < 1.0:
                return librosa.resample(y, orig_sr=audio.SR, target_sr=int(audio.SR / 2 ** (tom / 12)))
            if abs(veloc - 1) > 0.02:
                y = librosa.effects.time_stretch(y, rate=veloc)
            if abs(tom) > 0.3:
                y = librosa.effects.pitch_shift(y, sr=audio.SR, n_steps=tom)
            return y

        gravar(os.path.join(pasta, "0_original.wav"), mix)
        gravar(os.path.join(pasta, "1_lofi_real.wav"), lofi)
        # A voz fica limpa: só desacelera junto com o resto. Quem vira lo-fi é a base.
        v = lento(voz)
        i = lento(tratar(inst, m_inst, dev))
        juntar = lambda *partes: sum(nivel(p[: min(map(len, [x for x, _ in partes]))], db) for p, db in partes)
        gravar(os.path.join(pasta, "2_voz_limpa_inst_unet.wav"), juntar((v, -20), (i, -22)))
        if gerador is not None:
            i_l = lento(inst)
            bpm = andamento(i_l, audio.SR)
            bpm = bpm / 2 if bpm > 110 else bpm
            base = gerador.gerar(i_l, audio.SR, bpm)
            gravar(os.path.join(pasta, "3_voz_limpa_musicgen.wav"), juntar((v, -20), (base, -23)))
            gravar(os.path.join(pasta, "4_voz_limpa_musicgen_inst.wav"), juntar((v, -20), (base, -24), (i, -27)))
        torch.cuda.empty_cache()
    pagina()


ROTULOS = {
    "0_original": "Original",
    "1_lofi_real": "Lo-fi de verdade (do YouTube)",
    "2_voz_limpa_inst_unet": "Voz original + instrumental com timbre lo-fi (U-Net)",
    "3_voz_limpa_musicgen": "Voz original + batida lo-fi nova (MusicGen, sincronizada)",
    "4_voz_limpa_musicgen_inst": "Voz original + batida nova + um pouco do instrumental lo-fi",
}


def pagina(pasta=os.path.join("exemplos", "finais")):
    """Página local com um player para cada versão, para comparar de fone."""
    import html
    secoes = []
    for m in sorted(d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d))):
        faixas = "".join(
            f'<div class="faixa"><span>{html.escape(r)}</span>'
            f'<audio controls preload="none" src="{m}/{f}.wav"></audio></div>'
            for f, r in ROTULOS.items() if os.path.exists(os.path.join(pasta, m, f + ".wav")))
        secoes.append(f"<section><h2>{html.escape(m.replace('_', ' '))}</h2>{faixas}</section>")
    estilo = (":root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--linha:#e3e2dc}"
              "@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mut:#9b9a94;--linha:#2c2c2a}}"
              "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:760px;"
              "margin:32px auto;padding:0 16px}h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:0 0 24px}"
              "h2{font-size:17px;margin:28px 0 8px}.faixa{display:flex;align-items:center;gap:12px;padding:8px 0;"
              "border-top:1px solid var(--linha)}.faixa span{flex:1;min-width:0}audio{width:300px;max-width:55vw;height:36px}")
    with open(os.path.join(pasta, "index.html"), "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Music ANN: comparação</title>'
                f"<style>{estilo}</style></head><body><h1>Music ANN: comparação de ouvido</h1>"
                f"<p>Trechos de músicas de teste, que nenhum modelo viu no treino. Ouça de fone, na ordem.</p>"
                f"{''.join(secoes)}</body></html>")
    print("página:", os.path.join(pasta, "index.html"))


if __name__ == "__main__":
    main()
