# Predição de acidentes fatais em rodovias federais

Modelo de classificação binária que estima se um acidente registrado pela **Polícia
Rodoviária Federal** resulta ou não em vítima fatal, a partir das características do
acidente: condições da via, clima, horário, tipo de ocorrência e quantidade de pessoas e
veículos envolvidos.

O trabalho segue a metodologia **CRISP-DM** (compreensão do negócio, compreensão dos dados,
preparação, modelagem e avaliação) e termina com uma avaliação do modelo na **proporção real**
de acidentes fatais, que é onde ele mostra para que serve de verdade.

---

## Principais resultados

- **XGBoost** foi o melhor de quatro modelos (AUC-ROC 0,827), com a regressão logística
  logo atrás (0,815).
- Na **proporção real** de acidentes fatais (~6,5%), a precisão cai de ~70% para **~15%**
  com recall de 0,80: o modelo não serve como um "sim ou não" isolado.
- Como **ranqueador de risco**, ele funciona: os **10% de acidentes com maior risco previsto
  concentram ~45% dos fatais**, um ganho de 4,6 vezes sobre a ordem aleatória.

---

## Dados

[Dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf),
arquivos de acidentes agrupados por ocorrência (`datatran{ano}.csv`), de **2018 a 2025**:
**543.100 acidentes**, dos quais apenas **7%** tiveram vítima fatal.

Para o treino, a base foi **balanceada** (todos os acidentes fatais + o mesmo número de não
fatais sorteados ao acaso, ano a ano). Depois da remoção de nulos e de outliers (mantendo
acidentes com até 5 pessoas e 5 veículos), ficaram **70.290 registros e 13 variáveis**, com
divisão estratificada de 80/20 entre treino e teste.

---

## Resultados no teste balanceado

Quatro modelos foram treinados e comparados nas mesmas divisões de dados (limiar padrão de 0,5):

| Modelo | Acurácia | Recall (fatal) | F1 (fatal) | AUC-ROC |
|---|---|---|---|---|
| **XGBoost** | **0,75** | **0,70** | **0,73** | **0,827** |
| Regressão Logística | 0,73 | 0,68 | 0,71 | 0,815 |
| Random Forest | 0,73 | 0,68 | 0,71 | 0,811 |
| Árvore de Decisão | 0,71 | 0,66 | 0,69 | 0,784 |

A regressão logística fica perto do XGBoost, sinal de que boa parte do sinal é linearmente
separável e de que o ganho do modelo mais complexo vem das interações entre condições da via e
tipo de acidente. LightGBM e CatBoost também foram testados e ficaram no mesmo patamar: o limite
parece estar nos **dados**, não no algoritmo.

As variáveis mais importantes são o **tipo de acidente** (atropelamento e colisão frontal
passam de 80% de fatalidade na base balanceada), a **causa** e a **fase do dia**.

Como deixar passar um acidente fatal é o erro mais grave, o limiar de decisão foi reduzido: em
0,35, o recall do XGBoost sobe para **0,84**, com precisão de 0,68 no teste balanceado.

---

## Avaliação na proporção real

O balanceamento foi feito **antes** da divisão treino/teste, então as métricas acima são
calculadas num teste com 48% de fatais, enquanto a realidade tem ~6,5%. O recall e a AUC-ROC
não dependem dessa proporção, mas a **precisão** depende muito.

A seção 6 do notebook final corrige isso:

1. **Estima a proporção real** de fatais no mesmo recorte do treino (6,54%).
2. **Repondera o teste:** cada acidente não fatal passa a pesar 13,2, o que é válido porque os
   não fatais da base balanceada foram sorteados aleatoriamente.
3. **Escolhe o limiar num conjunto de validação**, e não no teste, buscando recall ≥ 0,80.
4. **Refaz o pré-processamento num `Pipeline`**, para que o agrupamento de categorias raras e o
   One-Hot Encoding sejam aprendidos só no treino.

| Modelo | AUC-ROC | PR-AUC real | Recall | Precisão balanceada | Precisão real |
|---|---|---|---|---|---|
| **XGBoost** | **0,826** | **0,297** | 0,80 | 0,70 | **0,15** |
| Regressão Logística | 0,814 | 0,276 | 0,79 | 0,69 | 0,14 |

A referência de um modelo aleatório é PR-AUC = 0,065, então o modelo tem sinal real (4,5
vezes acima do acaso). Mas, com recall de 0,80, **só 1 em cada 7 alertas é um acidente
fatal**, e o modelo marcaria como fatal cerca de um terço das ocorrências.

O uso que faz sentido é o de **ranqueamento**, por exemplo para priorizar o atendimento de
ocorrências à medida que são registradas:

| Acidentes priorizados (maior risco primeiro) | Fatais capturados | Ganho sobre o acaso |
|---|---|---|
| 5% | 28,2% | 5,6x |
| 10% | 45,5% | 4,6x |
| 20% | 64,5% | 3,2x |
| 30% | 76,3% | 2,5x |

---

## Estrutura

Execute os notebooks nesta ordem para reproduzir do zero:

| Arquivo | Conteúdo |
|---|---|
| `MontagemDF.ipynb` | Lê os CSVs anuais da PRF, binariza o alvo e balanceia as classes ano a ano |
| `TratamentoDadosPRF.ipynb` | Análise exploratória da base, remoção de nulos e de outliers; gera o `prf_clean.parquet` |
| `Projeto_Final_PRF.ipynb` | Projeto nas fases do CRISP-DM: EDA, preparação, modelagem, avaliação e avaliação na proporção real |
| `data/prf_clean.parquet` | Dataset tratado e pronto para modelagem |

Os notebooks já vêm com as saídas executadas, então dá para ler os resultados sem rodar nada.
Os dois primeiros esperam os CSVs brutos da PRF numa pasta do Google Drive (ajuste a variável
`pasta` se for rodar em outro lugar). O `Projeto_Final_PRF.ipynb` só precisa do
`data/prf_clean.parquet` e roda localmente, no Colab ou no Kaggle.

---

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook Projeto_Final_PRF.ipynb
```

---

## Decisões técnicas

**Parquet em vez de CSV para os dados intermediários.** São oito anos de dados; o Parquet
preserva os tipos entre notebooks e evita reinferir dtypes a cada carregamento.

**Quatro modelos em vez de um.** A comparação é o resultado, não um passo intermediário: sem a
regressão logística como linha de base não dá para dizer se o ganho do XGBoost justifica o
custo de interpretabilidade.

**Recall da classe fatal como métrica principal.** Num uso real, deixar de sinalizar um
acidente com potencial fatal custa mais caro que um alarme falso.

**Avaliar na proporção real, e não só no teste balanceado.** Balancear o treino ajuda o modelo
a aprender a classe rara, mas medir a precisão num teste balanceado superestima o quanto os
alertas são confiáveis. Reponderar o teste mostra o desempenho que o modelo teria em produção e
muda a conclusão sobre como usá-lo.

**Limiar escolhido na validação.** Escolher o limiar olhando o teste deixa o resultado
otimista; na seção 6 ele é definido num conjunto de validação separado.

---

## Limitações

- **Só informações registradas na ocorrência.** Fatores que pesam muito na fatalidade, como
  velocidade real, uso de cinto e condição do veículo, não estão na base.
- **Recorte de até 5 pessoas e 5 veículos.** Acidentes com ônibus, vans e engavetamentos
  grandes ficaram de fora, e o modelo não deve ser usado para eles.
- **Probabilidades infladas.** O modelo foi treinado com 48% de fatais; para exibir as
  probabilidades como "chance de ser fatal", é preciso corrigi-las para a proporção real.
  Para ordenar ocorrências, isso não faz diferença.
- **Divisão aleatória, não temporal.** Treino e teste misturam todos os anos. Uma validação
  treinando em 2018–2023 e testando em 2024–2025 mostraria se o modelo se mantém ao longo do
  tempo.

---

Projeto final da disciplina de Machine Learning, Bacharelado em Ciência da Computação (Unama).
Autor: [João Guilherme Alves de Almeida](https://github.com/JGuilherme-aa1603).
