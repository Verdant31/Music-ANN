<h1 align="center">Music ANN</h1>
<p align="center">Experimento: transformar uma música na sua versão lo-fi.</p>
<p align="center"><strong>Python</strong> · PyTorch · librosa · Demucs</p>

A ideia, de 2023, era ensinar uma rede neural a fazer o que um produtor faz ao criar a versão lo-fi de uma música: a partir do espectrograma de Mel da faixa original, prever o espectrograma da versão lo-fi. Em 2026 o projeto foi retomado, a ideia foi testada a sério com 79 pares de músicas reais e o resultado mudou o caminho: a versão que soa bem é montada como um produtor monta, e a rede neural ficou com as partes em que ela de fato ajuda.

## Gerar uma lo-fi

```powershell
.venv\Scripts\python produzir_lofi.py minha_musica.mp3
.venv\Scripts\python produzir_lofi.py minha_musica.mp3 --inicio 60 --duracao 30 --velocidade 0.95
```

O que `musicann/producao.py` faz:

1. **Separa** a música em voz, bateria, baixo e o resto com o Demucs (rede neural pré-treinada).
2. **Voz:** fica a original, limpa, com um reverb leve. Nas lo-fi de verdade a voz não é abafada.
3. **Bateria:** sai a original e entra um loop de bateria lo-fi real ([Lo-Fi Drums Dataset](https://github.com/patchbanks/Lo-Fi-Drums-Dataset), CC BY 4.0, WaivOps), colado compasso a compasso nas batidas da música. O encaixe é escolhido por medida: testo vários andamentos (inclusive os períodos que mais se repetem na bateria original), as 4 posições do início do compasso e defasagens de até 1/4 de batida, e fico com a combinação cujos ataques mais coincidem com os da bateria original. Se nenhuma coincide o suficiente (música com swing, quase sem bateria), fica a bateria original com tratamento lo-fi. Loops com um golpe muito mais forte que os outros (~1/3 do conjunto) são descartados, e a bateria passa por um limitador.
4. **Baixo e o resto:** passa-baixa (o "abafado"), saturação de fita, oscilação lenta de velocidade (wow/flutter) e sidechain no bumbo.
5. **Tudo:** 5% mais lento mantendo o tom da voz, um pouco menos de graves e chiado de vinil.

Os ajustes ficam em `producao.Ajustes`.

### Opções

```powershell
.venv\Scripts\python produzir_lofi.py minha_musica.mp3 --riff --teclado progressao --timbre gravado
```

- `--perfil`: `rap` (base refeita, o padrão), `brasileira` (base original filtrada) ou `pesada` (rap com 808 na frente: a gravação original mais lenta e escura, como a lo-fi mais ouvida de *Chill Bill*).
- `--riff`: troca o teclado/sample original por um piano elétrico novo. Com `--teclado`:
  - `gancho`: quando o sample se repete (*4r Da Squaw*: uma frase de 2 compassos, 33 vezes), transcreve a frase pela média de todas as repetições e a toca limpa, com variações por seção e a cada 8 compassos;
  - `progressao`: toca os acordes da própria música, tempo a tempo, trocando de acorde junto com o baixo. Um acorde por compasso errava músicas como *No Role Modelz*, que trocam num ritmo de 5 + 3 tempos.
- `--timbre gravado`: o piano é um Rhodes Mark I de 1977 gravado nota a nota (veja Instalar); sem as amostras, usa o sintetizado.
- `--bateria`: `loop`, `substituicao` ou `original`; por padrão, o do perfil.

O estudo que levou ao teclado em progressão está em `musicann/harmonia.py`: nas lo-fis de J. Cole do canal Ivy Station, o produtor não copia o sample. Ele toca acordes simples de 3 notas, segurados, com a frase se repetindo a cada 4 ou 8 compassos.

## O que foi testado e o que se aprendeu

**Dados.** `pares.csv` tem a gravação original e uma lo-fi popular do YouTube de 46 músicas de rap americano e música brasileira; `pares_bollywood.csv`, 33 lo-fi flips oficiais de Bollywood. Os áudios não vão para o repositório.

**Alinhamento.** A lo-fi costuma ser mais lenta, cortar trechos e às vezes estar em outro tom. `preparar_voz.py` separa a voz das duas versões e, para cada bloco de ~3 s da lo-fi, acha o trecho correspondente na original com DTW de subsequência sobre o chroma da voz, refinando depois quadro a quadro. Com a voz separada, 95% dos blocos ficam certos com margem de ±46 ms; alinhando a mixagem eram 37%.

**1. U-Net prevendo o espectrograma lo-fi** (`treinar.py`, a ideia original). Funciona em dados limpos: com lo-fi sintética (uma cadeia de efeitos aplicada à própria original), erra 61% menos que a original e 39% menos que um equalizador fixo, com saída nítida. Nos pares reais, ganha nos números (erro L1 0,140 contra 0,161 da original, em 6 músicas fora do treino), mas de ouvido só abafa tudo, inclusive a voz. O motivo: a única coisa que se repete em todos os pares é um escurecimento médio; a batida, o piano e o arranjo novos mudam de produtor para produtor e não dá para prevê-los a partir da original.

**2. Batida nova com MusicGen-melody** (`musicann/batida.py`). Segue os acordes, mas sai com outro groove, sem relação com o ritmo da música, mesmo esticada para o andamento certo. Descartado.

**3. Montagem como produtor** (`musicann/producao.py`). É a que soa bem. Os ajustes foram afinados de ouvido: voz no tom original, desaceleração leve, abafado só na base.

**4. Rede que aprende os ajustes da montagem** (`medir_ajustes.py`, `treinar_ajustes.py`). Para cada par real mede o que a lo-fi fez (velocidade, tom, corte de agudos, graves) e treina uma rede pequena para prever esses ajustes a partir da original. Na validação deixando uma música de fora, nem a rede nem uma regressão linear ganham de simplesmente usar a média (velocidade: erro 0,062 da rede contra 0,063 da média; tom: 0,60 contra 0,65 semitom, com poucas características de entrada). Com 62 músicas, a escolha do produtor parece não depender do som da original; seria preciso bem mais pares para testar de novo. Por isso a montagem usa ajustes fixos.

## Instalar

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install librosa soundfile scipy matplotlib demucs transformers accelerate
```

Os loops de bateria vão em `dados/loops/` (os 800 primeiros do Lo-Fi Drums Dataset já bastam). Para `--timbre gravado`, baixe o ZIP do [jRhodes3d](https://github.com/sfzinstruments/jlearman.jRhodes3d) (Jeff Learman, amostras CC BY 4.0, ~95 MB) e descompacte em `dados/instrumentos/`, ficando `dados/instrumentos/jlearman.jRhodes3d-master/`. Para refazer os experimentos, baixe os pares para `Musics/Normal/<nome>.mp3` e `Musics/Lofi/<nome>.mp3` e rode `separar.py`, `preparar_voz.py` e depois `treinar.py` ou `medir_ajustes.py` + `treinar_ajustes.py`. Outros scripts: `preparar_sintetico.py` (pares de controle), `conferir_alinhamento.py` (mede o alinhamento e grava um .wav estéreo para ouvir), `exemplos_producao.py` (página de comparação de ouvido).

Os notebooks `fillDataset.ipynb` e `main.ipynb` são a versão de 2023, mantidos como registro.

## Versões

As versões que ficaram boas ganham uma tag no git e uma pasta em `versoes/`, com o passo a passo para recriar o resultado e o hash dos arquivos para conferir. A primeira é a [`v3-peace`](versoes/v3/README.md), com a base refeita e a lo-fi de *Peace* aprovada.
