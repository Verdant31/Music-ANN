"""Seções da música (introdução, verso, refrão, ponte...) com o SongFormer.

SongFormer (ASLP-lab, 2025, CC BY 4.0) é um modelo aberto de análise de
estrutura musical: recebe o áudio e devolve trechos com início, fim e
rótulo. Nos testes do artigo acerta ~80% das fronteiras na base Harmonix
(o agrupamento por repetição que eu usava antes errava bastante em músicas
em que verso e refrão têm a mesma energia, como Blinding Lights).

O modelo roda local. O código dele vem do repositório no Hugging Face
(trust_remote_code) e só monta a rede e lê os arquivos de configuração.
"""
import os
import sys

import numpy as np
import soundfile as sf

_MODELO = None

# Rótulos do SongFormer -> papéis usados na montagem
PAPEL = {"intro": "intro", "verse": "verso", "pre-chorus": "verso", "prechorus": "verso",
         "chorus": "refrao", "bridge": "ponte", "inst": "instrumental", "instrumental": "instrumental",
         "outro": "final", "silence": "final", "end": "final"}


def _carregar():
    global _MODELO
    if _MODELO is None:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        pasta = snapshot_download("ASLP-lab/SongFormer", ignore_patterns=["SongFormer.pt", "SongFormer.safetensors"])
        if pasta not in sys.path:
            sys.path.append(pasta)
        os.environ["SONGFORMER_LOCAL_DIR"] = pasta
        # O transformers 5 monta o modelo no dispositivo "meta" dentro do
        # from_pretrained, e o SongFormer cria tensores no __init__ que não
        # aceitam isso. Então monto a classe direto e carrego os pesos à mão.
        from configuration_songformer import SongFormerConfig
        from modeling_songformer import SongFormerModel
        modelo = SongFormerModel(SongFormerConfig.from_pretrained(pasta))
        faltando, sobrando = modelo.load_state_dict(load_file(os.path.join(pasta, "model.safetensors")), strict=False)
        if faltando:
            print(f"SongFormer: {len(faltando)} pesos não encontrados (ex.: {faltando[:3]})")
        # O MuQ (parte do SongFormer) guarda a config do conformer num dict
        # simples, sem o campo de implementação de atenção que o transformers 5
        # exige; uso a implementação padrão ("eager").
        for m in modelo.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None and not hasattr(cfg, "_attn_implementation"):
                try:
                    setattr(cfg, "_attn_implementation", "eager")
                except Exception:
                    pass
        _MODELO = modelo.to("cuda").eval()
    return _MODELO


def secoes(audio, sr, pasta_tmp="dados"):
    """Lista de dicts {ini, fim, papel, rotulo} para um áudio mono ou estéreo."""
    modelo = _carregar()
    caminho = os.path.join(pasta_tmp, "_songformer_tmp.wav")
    sf.write(caminho, audio.T if audio.ndim == 2 else audio, sr)
    try:
        # Em 32 bits: em float16 os valores estouram e ele não acha nenhuma
        # fronteira; bfloat16 quebra o pós-processamento (numpy não aceita).
        bruto = modelo(caminho)
    finally:
        os.remove(caminho)
    saida = []
    for s in bruto:
        rotulo = str(s["label"]).lower()
        saida.append(dict(ini=float(s["start"]), fim=float(s["end"]), rotulo=rotulo, papel=PAPEL.get(rotulo, "verso")))
    return saida
