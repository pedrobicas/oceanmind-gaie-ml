# Relatório Técnico - GAIE

**Projeto:** OceanMind - Inteligência Oceânica e Previsão Climática via Dados Espaciais  
**Disciplina:** Generative AI For Engineering (GAIE)  
**Global Solution:** 2026 - Indústria Espacial: O Código que Move o Universo  
**Curso:** Engenharia de Software - 4º ano  
**Repositório:** https://github.com/pedrobicas/oceanmind-gaie-ml  
**Aplicação:** https://oceanmind-gaie-ml.streamlit.app

## 1. Integrantes

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassão | 98607 |
| Pedro Bicas | 99534 |

## 2. Contexto

O OceanMind é uma solução voltada ao monitoramento de riscos oceânicos e costeiros. A ideia é usar dados ambientais, climáticos e espaciais para indicar quando uma região marítima apresenta maior risco.

Nesta parte da Global Solution, o foco foi criar um pipeline de Machine Learning para classificar o risco oceânico. A solução se conecta ao tema da Indústria Espacial porque trabalha com dados inspirados em observação terrestre e sensoriamento remoto, que são comuns em aplicações com satélites.

## 3. Objetivo

O objetivo do modelo é prever o campo:

```text
ocean_risk_level
```

As classes são:

- `baixo`
- `moderado`
- `alto`
- `critico`

## 4. Dados

Foi criado um dataset sintético com **1.600 registros e 15 colunas**. A geração foi feita em Python, no arquivo:

```text
src/generate_dataset.py
```

O CSV usado no treinamento está em:

```text
data/oceanmind_dataset.csv
```

As variáveis simulam condições oceânicas e climáticas, como temperatura da superfície do mar, anomalia térmica, vento, ondas, umidade, pressão atmosférica, nuvens, clorofila e velocidade da corrente.

Distribuição das classes:

| Classe | Registros |
|---|---:|
| moderado | 431 |
| baixo | 394 |
| critico | 390 |
| alto | 385 |

## 5. Metodologia

O pipeline foi implementado com `scikit-learn`. A coluna `sample_id` foi removida por ser apenas um identificador, e a variável alvo foi separada dos atributos.

A base foi dividida em treino e teste com estratificação, mantendo a distribuição das classes. Para o pré-processamento, foram usadas duas etapas:

- `StandardScaler` para as variáveis numéricas;
- `OneHotEncoder` para a variável categórica `region`.

Depois disso, foram treinados dois modelos de classificação:

- Random Forest Classifier;
- Gradient Boosting Classifier.

Os modelos foram avaliados com acurácia, precision, recall, f1-score e matriz de confusão.

## 6. Resultados

Resultados obtidos no conjunto de teste:

| Modelo | Acurácia |
|---|---:|
| Random Forest | 80,625% |
| Gradient Boosting | 80,3125% |

O Random Forest teve a melhor acurácia e foi escolhido como modelo final.

Arquivos gerados:

```text
models/random_forest.joblib
models/gradient_boosting.joblib
models/best_model.joblib
reports/metrics.json
reports/figures/confusion_matrix_random_forest.png
reports/figures/confusion_matrix_gradient_boosting.png
```

## 7. SHAP

Foi usada a biblioteca SHAP para interpretar o modelo final. Como o problema possui mais de uma classe, a importância foi agregada pela média absoluta dos valores SHAP.

Principais variáveis:

| Posição | Variável | Importância SHAP |
|---:|---|---:|
| 1 | `temperature_anomaly_c` | 0.1471 |
| 2 | `sea_surface_temperature_c` | 0.0602 |
| 3 | `wave_height_m` | 0.0289 |
| 4 | `wind_speed_kmh` | 0.0221 |
| 5 | `pressure_hpa` | 0.0209 |

A variável mais importante foi a anomalia térmica. Isso é coerente com o problema, porque alterações anormais na temperatura do mar podem indicar instabilidade climática e risco ambiental maior.

Arquivos:

```text
reports/shap_ranking.json
reports/figures/shap_feature_importance.png
```

## 8. Aplicação

A aplicação foi desenvolvida em Streamlit, no arquivo:

```text
app.py
```

Ela permite escolher uma região, ajustar os indicadores ambientais e consultar o nível de risco previsto pelo modelo. Também mostra as métricas de treinamento e as variáveis mais importantes segundo SHAP.

Aplicação publicada:

```text
https://oceanmind-gaie-ml.streamlit.app
```

## 9. Como Reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train_models.py
python src/shap_analysis.py
streamlit run app.py
```

## 10. Conclusão

O projeto entregou um pipeline completo de Machine Learning para classificação de risco oceânico, com geração de dados, pré-processamento, comparação de modelos, avaliação, SHAP e aplicação publicada em Streamlit.

O resultado final ficou integrado ao tema da Global Solution por usar dados ambientais e espaciais simulados para apoiar decisões em regiões marítimas e costeiras.
