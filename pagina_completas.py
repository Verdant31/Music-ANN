"""Página para ouvir as lo-fi completas ao lado das originais.

Para cada música em exemplos/completas/<nome>.wav (gerada por
produzir_lofi.py), põe a original e a lo-fi lado a lado, com os áudios
embutidos (abre em qualquer navegador) e os players sincronizados: mexeu na
posição de um, o outro vai para o mesmo ponto da música.

    python pagina_completas.py Weeknd_Blinding_Lights Djavan_Oceano --titulo "lo-fi v5"
"""
import argparse
import base64
import html
import json
import os
import subprocess

from exemplos_producao import FFMPEG
from musicann.producao import Ajustes

PAPEIS = dict(intro="introdução", verso="verso", refrao="refrão", ponte="ponte", instrumental="instrumental", final="final")
ESTILO = (":root{--bg:#fafaf7;--fg:#1d1d1b;--mut:#6b6b66;--linha:#e3e2dc}"
          "@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--mut:#9b9a94;--linha:#2c2c2a}}"
          "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:760px;"
          "margin:32px auto;padding:0 16px}h1{font-size:22px;margin:0 0 4px}p{color:var(--mut);margin:0 0 24px}"
          "h2{font-size:17px;margin:28px 0 8px}.faixa{display:flex;align-items:center;gap:12px;padding:8px 0;"
          "border-top:1px solid var(--linha)}.faixa span{flex:1;min-width:0}small{color:var(--mut)}"
          "audio{width:300px;max-width:55vw;height:36px}")
SCRIPT = """<script>
// Os players de cada música andam juntos (a lo-fi é mais lenta, então o tempo
// é convertido). Só um toca por vez; ao trocar, continua do mesmo ponto. A
// posição só é copiada quando você mexe (pausa, arrasta ou dá play em outro):
// nada é reposicionado enquanto um deles toca.
const todos = [...document.querySelectorAll('audio')];
const grupo = a => todos.filter(o => o !== a && o.dataset.par === a.dataset.par);
const naMusica = a => a.currentTime / Number(a.dataset.fator);  // segundos na original
function mover(o, t) {
  const alvo = t * Number(o.dataset.fator);
  if (Math.abs(o.currentTime - alvo) < 0.05) return;
  o.dataset.auto = Date.now();
  try { o.currentTime = alvo; } catch (e) {}
}
const copiar = de => grupo(de).forEach(o => { if (o.paused) mover(o, naMusica(de)); });
todos.forEach(a => {
  a.volume = 0.5;
  a.addEventListener('seeked', () => { if (Date.now() - Number(a.dataset.auto || 0) > 1500) copiar(a); });
  a.addEventListener('pause', () => copiar(a));
  a.addEventListener('play', () => {
    const tocando = grupo(a).find(o => !o.paused);
    if (tocando) { const t = naMusica(tocando); tocando.pause(); mover(a, t); }
    todos.forEach(o => { if (o !== a && !o.paused) o.pause(); });
  });
  // Volume e mudo também andam juntos.
  a.addEventListener('volumechange', () => grupo(a).forEach(o => {
    if (Math.abs(o.volume - a.volume) > 0.001) o.volume = a.volume;
    if (o.muted !== a.muted) o.muted = a.muted;
  }));
});
</script>"""


def mp3_base64(origem, destino, kbps):
    subprocess.run([FFMPEG, "-loglevel", "error", "-y", "-i", origem, "-af", "loudnorm=I=-20:TP=-2:LRA=11",
                    "-ar", "44100", "-b:a", f"{kbps}k", destino], check=True)
    return base64.b64encode(open(destino, "rb").read()).decode()


def mapa_secoes(pasta, nome, base, veloc):
    """Seções que a montagem usou (já no tempo da lo-fi) ou, sem elas, as do
    SongFormer convertidas para o tempo da lo-fi."""
    usadas = os.path.join(pasta, f"{base}.secoes.json")
    arq = os.path.join("dados", "secoes", f"{nome}_0_0.json")
    if os.path.exists(usadas):
        s = json.load(open(usadas, encoding="utf-8"))
        return " · ".join(f"{int(t // 60)}:{int(t % 60):02d} {PAPEIS.get(p, p)}" for t, p in s)
    if os.path.exists(arq):
        s = json.load(open(arq, encoding="utf-8"))
        p = [x for i, x in enumerate(s) if i == 0 or x["papel"] != s[i - 1]["papel"]]
        return " · ".join(f"{int(x['ini'] / veloc // 60)}:{int(x['ini'] / veloc % 60):02d} {PAPEIS.get(x['papel'], x['papel'])}" for x in p)
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("musicas", nargs="+")
    ap.add_argument("--titulo", default="original e lo-fi")
    ap.add_argument("--descricao", default="")
    ap.add_argument("--kbps", type=int, default=128)
    ap.add_argument("--pasta", default=os.path.join("exemplos", "completas"))
    ap.add_argument("--saida", default="completas_com_originais.html")
    ap.add_argument("--versoes", nargs="+", default=[""],
                    help="sufixos das versões lo-fi de cada música (<nome><sufixo>.wav)")
    ap.add_argument("--rotulos", nargs="+", default=None, help="nome de cada versão na página")
    ap.add_argument("--referencia", nargs="+", default=[], metavar="MUSICA|ARQUIVO|ROTULO",
                    help="faixa de referência (por exemplo, uma lo-fi do YouTube) na seção da música, sem sincronizar")
    args = ap.parse_args()
    veloc = Ajustes().velocidade

    secoes_html = ""
    for k, nome in enumerate(args.musicas):
        orig = mp3_base64(os.path.join("Musics", "Normal", nome + ".mp3"), os.path.join(args.pasta, f"{nome}_original.mp3"), args.kbps)
        rotulos = args.rotulos or ["Lo-fi"] * len(args.versoes)
        faixas = ""
        for sufixo, rotulo in zip(args.versoes, rotulos):
            base = nome + sufixo
            if not os.path.exists(os.path.join(args.pasta, base + ".wav")):
                continue  # versão que essa música não tem
            lofi = mp3_base64(os.path.join(args.pasta, base + ".wav"), os.path.join(args.pasta, base + ".mp3"), args.kbps)
            arq_info = os.path.join(args.pasta, base + ".info.json")
            v = json.load(open(arq_info, encoding="utf-8"))["velocidade"] if os.path.exists(arq_info) else veloc
            faixas += (f'<div class="faixa"><span>{html.escape(rotulo)}<br><small>{html.escape(mapa_secoes(args.pasta, nome, base, v))}</small></span>'
                       f'<audio controls preload="none" data-par="{k}" data-fator="{1 / v:.6f}" '
                       f'src="data:audio/mpeg;base64,{lofi}"></audio></div>')
        for r, ref in enumerate(args.referencia):
            m_ref, arq_ref, rot_ref = ref.split("|")
            if m_ref != nome:
                continue
            dados = mp3_base64(arq_ref, os.path.join(args.pasta, f"{nome}_referencia{r}.mp3"), args.kbps)
            faixas = (f'<div class="faixa"><span>{html.escape(rot_ref)}<br><small>não anda junto com as outras</small></span>'
                      f'<audio controls preload="none" data-par="ref{k}_{r}" data-fator="1" '
                      f'src="data:audio/mpeg;base64,{dados}"></audio></div>') + faixas
        secoes_html += (
            f"<section><h2>{html.escape(nome.replace('_', ' '))}</h2>"
            f'<div class="faixa"><span>Original</span><audio controls preload="none" data-par="{k}" data-fator="1" '
            f'src="data:audio/mpeg;base64,{orig}"></audio></div>{faixas}</section>')
    caminho = os.path.join(args.pasta, args.saida)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Music ANN: {html.escape(args.titulo)}</title>'
                f"<style>{ESTILO}</style></head><body><h1>Music ANN: {html.escape(args.titulo)}</h1>"
                f"<p>Os players de cada música andam juntos: arrastou um, os outros vão para o mesmo ponto da música, "
                f"e o volume também anda junto. "
                f"{html.escape(args.descricao)}</p>{secoes_html}{SCRIPT}</body></html>")
    print(f"página: {caminho} ({os.path.getsize(caminho) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
