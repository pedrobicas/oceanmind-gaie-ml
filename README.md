# OceanMind - Machine Learning Pipeline (GAIE)

Entrega da disciplina **Generative AI For Engineering (GAIE)** para a **Global Solution 2026 - FIAP**.

## Identificação

**Projeto:** OceanMind - Inteligência Oceânica e Previsão Climática via Dados Espaciais  
**Tema da Global Solution:** Indústria Espacial: O Código que Move o Universo  
**Disciplina:** Generative AI For Engineering (GAIE)  
**Curso:** Engenharia de Software - 4º ano  
**Repositório GitHub:** https://github.com/pedrobicas/oceanmind-gaie-ml

### Integrantes

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassão | 98607 |
| Pedro Bicas | 99534 |

## Links

| Item | Link |
|---|---|
| Repositório | https://github.com/pedrobicas/oceanmind-gaie-ml |
| Aplicação | https://oceanmind-gaie-ml.streamlit.app |
| Execução local | http://localhost:8501 |

## Sobre o Projeto

O OceanMind é uma proposta de inteligência oceânica para monitorar riscos marítimos e costeiros usando dados espaciais, climáticos e ambientais.

Nesta entrega, o foco ficou na parte de Machine Learning: classificar o nível de risco oceânico/climático de uma região a partir de indicadores como temperatura do mar, anomalia térmica, vento, ondas, pressão atmosférica, nuvens e corrente marítima.

A conexão com a Indústria Espacial está no uso simulado de dados de observação terrestre e sensoriamento remoto, que poderiam vir de satélites ou plataformas climáticas.

## Objetivo do Modelo

O modelo prevê a variável:

```text
ocean_risk_level
```

Classes possíveis:

```text
baixo
moderado
alto
critico
```

## Dados

Foi usado um dataset sintético gerado em Python:

```text
src/generate_dataset.py
```

Arquivo final:

```text
data/oceanmind_dataset.csv
```

Resumo da base:

| Item | Valor |
|---|---:|
| Registros | 1.600 |
| Colunas | 15 |
| Variáveis usadas no modelo | 13 |
| Variável alvo | `ocean_risk_level` |

Distribuição das classes:

| Classe | Registros |
|---|---:|
| moderado | 431 |
| baixo | 394 |
| critico | 390 |
| alto | 385 |

Algumas variáveis usadas:

- região monitorada;
- latitude e longitude;
- mês;
- temperatura da superfície do mar;
- anomalia térmica;
- velocidade do vento;
- altura das ondas;
- umidade;
- pressão atmosférica;
- cobertura de nuvens;
- índice de clorofila;
- velocidade da corrente.

## Pipeline

O pipeline foi feito com `scikit-learn` e contempla:

1. carregamento do CSV;
2. separação entre atributos e alvo;
3. remoção da coluna `sample_id`;
4. divisão treino/teste com estratificação;
5. normalização das variáveis numéricas;
6. One-Hot Encoding da variável `region`;
7. treino de dois modelos;
8. comparação por métricas;
9. escolha do melhor modelo;
10. análise SHAP;
11. app Streamlit para demonstração.

## Modelos

Foram testados:

| Modelo | Arquivo |
|---|---|
| Random Forest Classifier | `models/random_forest.joblib` |
| Gradient Boosting Classifier | `models/gradient_boosting.joblib` |

O melhor modelo fica salvo em:

```text
models/best_model.joblib
```

## Resultados

Treinamento validado localmente em 29/05/2026.

| Modelo | Acurácia |
|---|---:|
| Random Forest | 80,625% |
| Gradient Boosting | 80,3125% |

Melhor modelo:

```text
random_forest
```

Arquivos de saída:

```text
reports/metrics.json
reports/figures/confusion_matrix_random_forest.png
reports/figures/confusion_matrix_gradient_boosting.png
```

## SHAP

A análise SHAP foi usada para entender quais variáveis mais pesaram nas previsões.

| Posição | Variável | Importância SHAP |
|---:|---|---:|
| 1 | `temperature_anomaly_c` | 0.1471 |
| 2 | `sea_surface_temperature_c` | 0.0602 |
| 3 | `wave_height_m` | 0.0289 |
| 4 | `wind_speed_kmh` | 0.0221 |
| 5 | `pressure_hpa` | 0.0209 |

A anomalia térmica foi a variável mais forte, o que faz sentido para o problema, já que mudanças anormais na temperatura do mar são um sinal importante de risco ambiental e climático.

Arquivos:

```text
reports/figures/shap_feature_importance.png
reports/shap_ranking.json
```

## Aplicação

A aplicação foi feita em Streamlit e permite simular uma região, ajustar os indicadores ambientais e ver o risco previsto.

Link:

```text
https://oceanmind-gaie-ml.streamlit.app
```

## Como Executar Localmente

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative no Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Treine os modelos:

```bash
python src/train_models.py
```

Gere a análise SHAP:

```bash
python src/shap_analysis.py
```

Rode o app:

```bash
streamlit run app.py
```

Abra:

```text
http://localhost:8501
```

## Relatório

O relatório técnico está disponível em Markdown e PDF:

```text
reports/relatorio_gaie.md
reports/relatorio_gaie.pdf
```

## Estrutura

```text
app.py
requirements.txt
README.md
.streamlit/config.toml
data/oceanmind_dataset.csv
models/best_model.joblib
models/gradient_boosting.joblib
models/random_forest.joblib
notebooks/oceanmind_ml_pipeline.ipynb
reports/metrics.json
reports/relatorio_gaie.md
reports/shap_ranking.json
reports/figures/confusion_matrix_gradient_boosting.png
reports/figures/confusion_matrix_random_forest.png
reports/figures/shap_feature_importance.png
src/generate_dataset.py
src/predict.py
src/shap_analysis.py
src/train_models.py
```
