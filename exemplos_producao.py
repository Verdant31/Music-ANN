"""Trechos de comparação da lo-fi montada como produtor (musicann/producao.py).

Para cada música grava em exemplos/producao/<música>/ duas versões novas
(5% mais lenta e sem desacelerar) e copia a original e a lo-fi do YouTube de
exemplos/finais, e monta exemplos/producao/comparacao.html com os áudios
embutidos (abre em qualquer aparelho).

    python exemplos_producao.py Kendrick_Money_Trees Djavan_Oceano --inicio 60
"""
import argparse
import base64
import html
import os
import shutil
import subprocess

import librosa
import soundfile as sf

from musicann import producao

FFMPEG = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\yt-dlp.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-N-128251-g6d1bfd7609-win64-gpl\bin\ffmpeg.exe")
ROTULOS = {
    "0_original": "Original",
    "1_lofi_real": "Lo-fi de verdade (do YouTube)",
    "2_nova_a": "Nova lo-fi: 5% mais lenta, voz no tom original",
    "3_nova_b": "Nova lo-fi: sem desacelerar",
}
VERSOES = {"2_nova_a": dict(velocidade=0.95), "3_nova_b": dict(velocidade=1.0)}


def pagina(pasta, titulo):
    secoes = []
    for m in sorted(d for d in os.listdir(pasta) if os.path.isdir(os.path.join(pasta, d))):
        faixas = ""
        for f, r in ROTULOS.items():
            wav = os.path.join(pasta, m, f + ".wav")
            if not os.path.exists(wav):
                continue
            mp3 = wav[:-4] + ".mp3"
            subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", wav, "-af", "loudnorm=I=-24:TP=-3:LRA=11",
                            "-ar", "44100", "-b:a", "96k", mp3], check=True)
            b64 = base64.b64encode(open(mp3, "rb").read()).decode()
            faixas += (f'<div class="faixa"><span>{html.escape(r)}</span><audio controls preload="none" '
                       f'src="data:audio/mpeg;base64,{b64}"></audio></div>')
        secoes.append(f"<section><h2>{html.escape(m.replace('_', ' '))}</h2>{faixas}</section>")
    estilo = (":root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--linha:#e3e2dc}"
              "@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mut:#9b9a94;--linha:#2c2c2a}}"
              "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:760px;"
              "margin:32px auto;padding:0 16px}h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:0 0 24px}"
              "h2{font-size:17px;margin:28px 0 8px}.faixa{display:flex;align-items:center;gap:12px;padding:8px 0;"
              "border-top:1px solid var(--linha)}.faixa span{flex:1;min-width:0}audio{width:300px;max-width:55vw;height:36px}")
    script = ("<script>document.querySelectorAll('audio').forEach(a=>{a.volume=0.5;a.addEventListener('play',"
              "()=>document.querySelectorAll('audio').forEach(o=>{if(o!==a)o.pause()}))})</script>")
    caminho = os.path.join(pasta, "comparacao.html")
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Music ANN: lo-fi</title>'
                f"<style>{estilo}</style></head><body><h1>{html.escape(titulo)}</h1>"
                f"<p>Voz original limpa, bateria trocada por um loop lo-fi encaixado nas batidas, base com "
                f"timbre de fita e chiado de vinil. Ouça de fone.</p>{''.join(secoes)}{script}</body></html>")
    print("página:", caminho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("musicas", nargs="+")
    ap.add_argument("--inicio", type=float, default=60.0)
    ap.add_argument("--duracao", type=float, default=30.0)
    ap.add_argument("--saida", default=os.path.join("exemplos", "producao"))
    args = ap.parse_args()

    from demucs.pretrained import get_model
    from separar import separar4
    modelo = get_model("htdemucs").to("cuda").eval()
    for nome in args.musicas:
        pasta = os.path.join(args.saida, nome)
        os.makedirs(pasta, exist_ok=True)
        for f in ("0_original", "1_lofi_real"):
            antigo = os.path.join("exemplos", "finais", nome, f + ".wav")
            if os.path.exists(antigo):
                shutil.copy(antigo, os.path.join(pasta, f + ".wav"))
        y, _ = librosa.load(os.path.join("Musics", "Normal", nome + ".mp3"), sr=producao.SR, mono=True,
                            offset=args.inicio, duration=args.duracao)
        partes = separar4(modelo, y, producao.SR, "cuda")
        for f, ajustes in VERSOES.items():
            mix, info = producao.produzir(partes, producao.Ajustes(**ajustes), semente=0)
            sf.write(os.path.join(pasta, f + ".wav"), mix, producao.SR)
            print(nome, f, info)
    pagina(args.saida, "Music ANN: lo-fi montada como produtor")


if __name__ == "__main__":
    main()
