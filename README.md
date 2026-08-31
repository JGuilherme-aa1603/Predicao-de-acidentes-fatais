# Predição de acidentes fatais em rodovias federais

Modelo de classificação binária que prevê se um acidente registrado pela **Polícia
Rodoviária Federal** resulta ou não em vítima fatal, a partir das características do
acidente — condições da via, clima, horário, tipo de ocorrência e perfil dos envolvidos.

O trabalho segue integralmente a metodologia **CRISP-DM**: compreensão do negócio,
compreensão dos dados, preparação, modelagem e avaliação.

---

## Dados

Base pública de ocorrências da PRF, **2018 a 2025**. Depois do tratamento e da montagem
do dataframe final: **70.290 registros e 13 variáveis**, com divisão de 80/20 entre
treino e teste (14.058 registros no conjunto de teste).

O alvo é balanceado por construção — 7.310 casos não fatais e 6.748 fatais no teste —
o que torna a acurácia uma métrica legível, sem o efeito de classe majoritária.

---

## Resultados

Quatro modelos foram treinados e comparados nas mesmas divisões de dados:

| Modelo | Acurácia | Recall (fatal) | F1 (fatal) | AUC-ROC |
|---|---|---|---|---|
| **XGBoost** | **0,75** | **0,70** | **0,73** | **0,8275** |
| Regressão Logística | 0,73 | 0,68 | 0,71 | 0,8145 |
| Random Forest | 0,73 | 0,68 | 0,71 | 0,8110 |
| Árvore de Decisão | 0,71 | 0,66 | 0,69 | 0,7840 |

**XGBoost venceu em todas as métricas.** A regressão logística fica surpreendentemente
perto (AUC 0,8145 contra 0,8275) — sinal de que boa parte do sinal no conjunto é
linearmente separável, e que o ganho do modelo mais complexo vem das interações entre
condições da via e tipo de acidente.

O recall da classe fatal (0,70) é a métrica que mais importa aqui: num uso real, deixar
de sinalizar um acidente com potencial fatal custa mais caro que um alarme falso.

---

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `TratamentoDadosPRF.ipynb` | Limpeza dos CSVs anuais da PRF — tipos, valores ausentes, normalização de categorias |
| `MontagemDF.ipynb` | Consolidação dos anos num dataframe único e engenharia de atributos |
| `Projeto_Final_PRF.ipynb` | Projeto final nas cinco fases do CRISP-DM: EDA, preparação, modelagem e avaliação |
| `data/prf_clean.parquet` | Dataset consolidado e pronto para modelagem |

---

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost matplotlib seaborn pyarrow jupyter

jupyter notebook Projeto_Final_PRF.ipynb
```

Os notebooks são independentes e já vêm com as saídas executadas — dá para ler os
resultados sem rodar nada. Para reproduzir do zero, execute-os na ordem da tabela acima.

---

## Decisões técnicas

**Parquet em vez de CSV para o dataset consolidado.** São oito anos de dados; o Parquet
preserva os tipos entre notebooks e evita o retrabalho de reinferir dtypes a cada
carregamento.

**Quatro modelos em vez de um.** A comparação é o resultado, não um passo intermediário:
sem a regressão logística como linha de base não dá para dizer se o ganho do XGBoost
justifica o custo de interpretabilidade.

**AUC-ROC como critério de desempate.** Acurácia sozinha não distingue modelos que
erram de formas diferentes; a AUC mede a qualidade do ranqueamento independentemente do
limiar de decisão escolhido.

---

Projeto final da disciplina de Machine Learning.
