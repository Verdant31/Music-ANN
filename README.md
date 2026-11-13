<h1 align="center">Music ANN</h1>
<p align="center">Experimento: transformar uma música na sua versão lo-fi com uma rede neural.</p>
<p align="center"><strong>Python</strong> · librosa · TensorFlow/Keras · pandas</p>

> [!NOTE]
> Protótipo inacabado, de 2023. A preparação dos dados está pronta; o treino da rede não chegou a rodar.

A ideia é ensinar uma rede neural a fazer o que um produtor faz à mão ao criar a versão lo-fi de uma música: a partir do espectrograma da faixa original, gerar o espectrograma da versão lo-fi.

## Como funciona

1. **Pares de faixas.** `Musics/Normal` e `Musics/Lofi` guardam a mesma música nas duas versões, com o mesmo nome de arquivo. Hoje são três pares.
2. **Espectrogramas.** `fillDataset.ipynb` carrega cada faixa com o librosa e calcula o espectrograma de Mel, com 128 bandas, janela de 2.048 amostras e passo de 512. Todos são cortados ou completados para 8.509 quadros, cerca de 3 min 18 s de áudio, e cada par vira uma linha de `mel_spectrogram_pairs.csv`.
3. **Rede.** `main.ipynb` lê o CSV, achata o espectrograma da versão original e define uma rede densa (250 → 150 → 50 → 110 → 80 → 8.509) para prever a versão lo-fi.

## Onde parou

A rede espera 8.509 valores de entrada, mas cada espectrograma achatado tem 128 × 8.509 = 1.089.152. A saída usa softmax com entropia cruzada, que serve para classificação, e não para gerar valores contínuos. Por isso o treino não roda como está.

Para continuar:

- Tratar o problema como regressão de imagem para imagem, com uma rede convolucional do tipo U-Net e perda MSE ou L1.
- Treinar com trechos curtos do espectrograma em vez da faixa inteira, o que também multiplica o número de exemplos.
- Reunir muito mais pares de faixas.
- Converter o espectrograma gerado de volta em áudio, com Griffin-Lim ou um vocoder neural.

## Instalar

```powershell
pip install librosa pandas numpy tensorflow notebook
```

Os caminhos das pastas de música em `fillDataset.ipynb` apontam para `D:/music-ann` e precisam ser ajustados. O CSV gerado está no `.gitignore`.
