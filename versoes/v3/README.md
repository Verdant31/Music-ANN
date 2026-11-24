# Versão v3: base refeita

A tag `v3-peace` guarda o código exato que gerou a lo-fi de *Peace* (Travis Thompson) aprovada em 24/11/2026. É o ponto seguro para voltar se uma mudança futura piorar o resultado.

## O que esta versão faz

- **Voz:** fica clara, com um toque de fita.
- **Buracos da voz:** são tapados nos golpes fortes da bateria (`musicann/reparo.py`).
- **Bateria:** loop lo-fi encaixado golpe a golpe, com os pratos da bateria original por cima.
- **Teclado:** os acordes da música num piano elétrico com sétima e nona no tom (`musicann/harmonia.py`), no lugar do teclado e dos samples originais.
- **Baixo:** um baixo novo, colado no bumbo do loop.
- **Equilíbrio:** a base segue o equilíbrio da música original, 2 dB mais forte em relação à voz.
- **Velocidade:** 0,95, como um disco mais lento, com chiado de vinil por cima.
- **Seções:** vêm do SongFormer.

Essas escolhas saíram de `estudar_lofis.py`, que comparou 29 remixes lo-fi do YouTube com as músicas originais.

## Como recriar a Peace aprovada

```powershell
git checkout v3-peace
yt-dlp -x --audio-format mp3 --audio-quality 0 -o "Musics/Normal/TravisThompson_Peace.%(ext)s" https://www.youtube.com/watch?v=uuhcill_wn4
New-Item -ItemType Directory -Force dados\secoes
Copy-Item versoes\v3\secoes\*.json dados\secoes\
python produzir_lofi.py Musics/Normal/TravisThompson_Peace.mp3 --saida TravisThompson_Peace.wav
```

### O que mais é preciso

- **Loops de bateria:** os 800 primeiros do [Lo-Fi Drums Dataset](https://github.com/patchbanks/Lo-Fi-Drums-Dataset) (CC BY 4.0) em `dados/loops/`. A lista exata está em `loops.json`.
- **Bibliotecas:** as versões usadas estão em `requirements.txt`, com Python 3.14, torch 2.14.1+cu130 e demucs 4.1.0 (modelo `htdemucs`).
- **Seções:** as do SongFormer estão em `secoes/`. Copiadas para `dados/secoes/`, o SongFormer nem precisa rodar.

### Conferência (SHA-256)

| Arquivo | SHA-256 |
|---|---|
| Entrada (`TravisThompson_Peace.mp3`) | `9ad6ccbd01feee186582dea903979a43b06d9d14025dc4010a6f98da16a7e7e6` |
| Partes separadas (`dados/stems/TravisThompson_Peace_htdemucs.npz`) | `170802fdd352774a42c5de8e4017ad1933072397606d213d203d286073dd2de4` |
| Saída aprovada (`TravisThompson_Peace.wav`) | `96f5695f711235d29e050718094b5156477fff6a21e30d5b127d9a12a313bc06` |

Com as mesmas partes separadas, a saída sai idêntica, bit a bit: o código desta tag foi conferido assim. Uma separação nova pelo Demucs, na GPU, pode variar em casas decimais muito pequenas. Nesse caso a saída não sai idêntica, mas soa igual.

O áudio não fica no repositório. Na máquina onde a versão foi aprovada, a entrada, as partes separadas e a saída estão em `exemplos/aprovadas/v3-peace/`, uma pasta fora do git.
