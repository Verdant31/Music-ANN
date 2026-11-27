<h1 align="center">Music ANN</h1>
<p align="center">Experimento: transformar uma música na sua versão lo-fi.</p>
<p align="center"><strong>Python</strong> · PyTorch · librosa · Demucs</p>

Começou em 2023 com uma ideia simples: treinar uma rede neural que recebe o espectrograma de Mel de uma música e devolve o espectrograma da versão lo-fi. Em 2026 retomei o projeto e testei isso com 79 pares de músicas reais (original e lo-fi do YouTube). A rede sozinha não chegou num resultado bom, então a versão atual monta a lo-fi como um produtor faria e usa redes neurais só onde elas ajudam, como na separação das faixas.

## Gerar uma lo-fi

```powershell
.venv\Scripts\python produzir_lofi.py minha_musica.mp3
.venv\Scripts\python produzir_lofi.py minha_musica.mp3 --inicio 60 --duracao 30 --velocidade 0.95
```

O que `musicann/producao.py` faz:

1. Separa a música em voz, bateria, baixo e o resto com o Demucs.
2. Mantém a voz original, limpa, com um reverb leve. Nas lo-fi do YouTube a voz quase nunca é abafada.
3. Tira a bateria original e coloca um loop de bateria lo-fi ([Lo-Fi Drums Dataset](https://github.com/patchbanks/Lo-Fi-Drums-Dataset), CC BY 4.0, WaivOps), colado compasso a compasso nas batidas da música. Para escolher o encaixe, testo vários andamentos, as 4 posições de início do compasso e defasagens de até 1/4 de batida, e fico com a combinação em que os golpes do loop mais coincidem com os da bateria original. Se nada encaixa (música com swing, quase sem bateria), fica a bateria original com tratamento lo-fi. Loops com um golpe muito mais forte que os outros são descartados, e a bateria passa por um limitador.
4. No baixo e no resto entram passa-baixa, saturação de fita, oscilação lenta de velocidade (wow/flutter) e sidechain no bumbo.
5. A mixagem final fica 5% mais lenta, com um pouco menos de grave e chiado de vinil.

Os parâmetros ficam em `producao.Ajustes`.

### Opções

```powershell
.venv\Scripts\python produzir_lofi.py minha_musica.mp3 --riff --teclado progressao --timbre gravado
```

- `--perfil`: `rap` (base refeita, o padrão), `brasileira` (base original filtrada) ou `pesada`, para rap com 808 alto (a gravação original mais lenta e escura).
- `--riff`: troca o teclado ou sample original por um piano elétrico novo. Com `--teclado` dá para escolher como:
  - `gancho`: quando o sample se repete a música inteira, transcreve a frase pela média de todas as repetições e toca ela limpa, com pequenas variações por seção e a cada 8 compassos.
  - `progressao`: toca os acordes da própria música, tempo a tempo, trocando de acorde junto com o baixo. Um acorde por compasso erra em músicas como *No Role Modelz*, que trocam num ritmo de 5 + 3 tempos.
- `--timbre gravado`: usa amostras de um Rhodes Mark I de 1977 (veja Instalar). Sem as amostras, usa o piano sintetizado.
- `--bateria`: `loop`, `substituicao` ou `original`. O padrão vem do perfil.

A ideia do teclado em progressão veio de comparar lo-fis de J. Cole do canal Ivy Station com as originais: o produtor não copia o sample, toca acordes simples de 3 notas, segurados, e a frase se repete a cada 4 ou 8 compassos.

## O que foi testado

`pares.csv` lista 46 músicas de rap americano e música brasileira, cada uma com a gravação original e uma lo-fi popular do YouTube. `pares_bollywood.csv` tem mais 33 lo-fi flips oficiais de Bollywood. Os áudios não estão no repositório.

Para comparar as duas versões é preciso alinhá-las, porque a lo-fi costuma ser mais lenta, cortar trechos e às vezes mudar de tom. `preparar_voz.py` separa a voz das duas e, para cada bloco de ~3 s da lo-fi, acha o trecho correspondente na original com DTW de subsequência sobre o chroma da voz, refinando depois quadro a quadro. Alinhando pela voz, 95% dos blocos ficam certos com margem de ±46 ms. Alinhando a mixagem inteira eram 37%.

1. U-Net prevendo o espectrograma lo-fi (`treinar.py`, a ideia original). Com lo-fi sintética (uma cadeia de efeitos aplicada à própria original) funciona bem: erra 61% menos que a original e 39% menos que um equalizador fixo. Nos pares reais ganha nos números (erro L1 0,140 contra 0,161 da original, em 6 músicas fora do treino), mas ouvindo, ela só abafa tudo, inclusive a voz. A única mudança que aparece em todos os pares é um escurecimento médio. A batida, o piano e o arranjo novos dependem do produtor e não dá para prevê-los a partir da original.
2. Batida nova com MusicGen-melody (`musicann/batida.py`). Segue os acordes, mas o groove não tem relação com o ritmo da música, mesmo esticado para o andamento certo. Descartei.
3. Montagem como produtor (`musicann/producao.py`). É a que soa bem, com os ajustes afinados de ouvido.
4. Rede que aprende os ajustes da montagem (`medir_ajustes.py`, `treinar_ajustes.py`). Para cada par mede o que a lo-fi fez (velocidade, tom, corte de agudos, graves) e treina uma rede pequena para prever isso a partir da original. Deixando uma música de fora na validação, nem a rede nem uma regressão linear ganham de usar a média (velocidade: erro 0,062 contra 0,063; tom: 0,60 contra 0,65 semitom). Com 62 músicas, a escolha do produtor não parece depender do som da original. Por isso a montagem usa ajustes fixos.

## Instalar

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install librosa soundfile scipy matplotlib demucs transformers accelerate
```

Os loops de bateria vão em `dados/loops/` (os 800 primeiros do Lo-Fi Drums Dataset bastam). Para `--timbre gravado`, baixe o ZIP do [jRhodes3d](https://github.com/sfzinstruments/jlearman.jRhodes3d) (Jeff Learman, amostras CC BY 4.0, ~95 MB) e descompacte em `dados/instrumentos/`, ficando `dados/instrumentos/jlearman.jRhodes3d-master/`.

Para refazer os experimentos, baixe os pares para `Musics/Normal/<nome>.mp3` e `Musics/Lofi/<nome>.mp3` e rode `separar.py`, `preparar_voz.py` e depois `treinar.py` ou `medir_ajustes.py` + `treinar_ajustes.py`. Outros scripts: `preparar_sintetico.py` (pares de controle), `conferir_alinhamento.py` (mede o alinhamento e grava um .wav estéreo para ouvir) e `exemplos_producao.py` (página para comparar versões).

Os notebooks `fillDataset.ipynb` e `main.ipynb` são a versão de 2023.

## Versões

Quando uma versão fica boa, ela ganha uma tag no git e uma pasta em `versoes/` com o passo a passo e os hashes para recriar o resultado. A primeira é a [`v3-peace`](versoes/v3/README.md), com a base refeita e a lo-fi de *Peace* como referência.

## Licença

O código está sob a [licença MIT](LICENSE). As músicas usadas nos testes, os loops de bateria e as amostras de Rhodes têm licenças próprias e não fazem parte do repositório.
