# Versão v3: base refeita

A tag `v3-peace` guarda o código que gerou a lo-fi de referência de *Peace* (Travis Thompson). Se alguma mudança piorar o resultado, dá para voltar para ela.

## O que muda nesta versão

- A voz fica clara, só com um toque de fita.
- Os buracos que a separação deixa na voz a cada golpe forte da bateria são tapados (`musicann/reparo.py`).
- A bateria é um loop lo-fi encaixado golpe a golpe, com os pratos da bateria original por cima.
- O teclado e os samples originais saem e entram os acordes da música num piano elétrico, com sétima e nona no tom (`musicann/harmonia.py`).
- Um baixo novo acompanha o bumbo do loop.
- A base segue o equilíbrio da música original, 2 dB mais forte em relação à voz.
- Velocidade 0,95, como um disco mais lento, com chiado de vinil.
- As seções vêm do SongFormer.

Essas escolhas vieram de `estudar_lofis.py`, que comparou 29 remixes lo-fi do YouTube com as músicas originais.

## Como recriar a lo-fi de Peace

```powershell
git checkout v3-peace
yt-dlp -x --audio-format mp3 --audio-quality 0 -o "Musics/Normal/TravisThompson_Peace.%(ext)s" https://www.youtube.com/watch?v=uuhcill_wn4
New-Item -ItemType Directory -Force dados\secoes
Copy-Item versoes\v3\secoes\*.json dados\secoes\
python produzir_lofi.py Musics/Normal/TravisThompson_Peace.mp3 --saida TravisThompson_Peace.wav
```

Também é preciso:

- os 800 primeiros loops do [Lo-Fi Drums Dataset](https://github.com/patchbanks/Lo-Fi-Drums-Dataset) (CC BY 4.0) em `dados/loops/` (a lista exata está em `loops.json`);
- as bibliotecas de `requirements.txt`, com Python 3.14, torch 2.14.1+cu130 e demucs 4.1.0 (modelo `htdemucs`);
- as seções de `secoes/` copiadas para `dados/secoes/`, para não precisar rodar o SongFormer.

### Conferência (SHA-256)

| Arquivo | SHA-256 |
|---|---|
| Entrada (`TravisThompson_Peace.mp3`) | `9ad6ccbd01feee186582dea903979a43b06d9d14025dc4010a6f98da16a7e7e6` |
| Partes separadas (`dados/stems/TravisThompson_Peace_htdemucs.npz`) | `170802fdd352774a42c5de8e4017ad1933072397606d213d203d286073dd2de4` |
| Saída (`TravisThompson_Peace.wav`) | `96f5695f711235d29e050718094b5156477fff6a21e30d5b127d9a12a313bc06` |

Com as mesmas partes separadas, a saída sai idêntica bit a bit. Se o Demucs separar de novo na GPU, os valores podem variar nas últimas casas decimais; aí a saída não bate no hash, mas soa igual.

O áudio não está no repositório.
