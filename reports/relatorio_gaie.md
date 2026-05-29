# Relatório Técnico - GAIE

**Projeto:** OceanMind - Inteligência Oceânica e Previsão Climática via Dados Espaciais  
**Disciplina:** Generative AI For Engineering (GAIE)  
**Global Solution:** 2026 - Indústria Espacial: O Código que Move o Universo  
**Curso:** Engenharia de Software - 4º ano  
**Repositório GitHub:** https://github.com/pedrobicas/oceanmind-gaie-ml  
**Aplicação publicada:** 

## 1. Integrantes

| Nome | RM |
|---|---:|
| Bryan Willian | 551305 |
| Felipe Terra | 99405 |
| Gabriel Doms | 98630 |
| Lucas Vassão | 98607 |
| Pedro Bicas | 99534 |

## 2. Contexto do Problema

O OceanMind é uma solução de inteligência oceânica voltada ao monitoramento de riscos marítimos e costeiros. A proposta utiliza dados espaciais, climáticos e ambientais para classificar o nível de risco de regiões oceânicas e apoiar a tomada de decisão por órgãos ambientais, pesquisadores, cidades costeiras e operações marítimas.

A conexão com a Indústria Espacial ocorre pelo uso simulado de dados de observação terrestre e sensoriamento remoto. Na prática, esse tipo de dado pode ser obtido por satélites, sensores orbitais e plataformas climáticas, permitindo monitorar variações ambientais em larga escala.

O problema de Machine Learning abordado é a classificação do nível de risco oceânico a partir de variáveis como temperatura da superfície do mar, anomalia térmica, velocidade do vento, altura das ondas, umidade, pressão atmosférica, cobertura de nuvens, clorofila e velocidade de correntes.

## 3. Objetivo da Solução

O objetivo da entrega GAIE é construir, treinar, avaliar, interpretar e demonstrar um pipeline completo de Machine Learning para prever a variável:

```text
ocean_risk_level
```

As classes previstas são:

- `baixo`
- `moderado`
- `alto`
- `critico`

## 4. Fonte e Qualidade dos Dados

Foi criado um dataset sintético com **1.600 registros e 15 colunas**, atendendo ao requisito mínimo da disciplina de pelo menos 1.000 linhas e 10 colunas. A geração dos dados está implementada em:

```text
src/generate_dataset.py
```

O dataset final está em:

```text
data/oceanmind_dataset.csv
```

As variáveis simulam medições ambientais e oceânicas em diferentes regiões, incluindo:

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

Distribuição das classes:

| Classe | Registros |
|---|---:|
| moderado | 431 |
| baixo | 394 |
| critico | 390 |
| alto | 385 |

A distribuição é equilibrada o suficiente para permitir treinamento e avaliação sem domínio extremo de uma única classe.

## 5. Metodologia

O pipeline foi implementado em Python com `scikit-learn`, utilizando `Pipeline` e `ColumnTransformer`. Essa escolha mantém o pré-processamento e o modelo integrados em um único objeto serializável, reduzindo risco de inconsistência entre treinamento e predição.

Etapas executadas:

1. Carregamento do dataset CSV.
2. Separação entre atributos e variável alvo.
3. Remoção da coluna identificadora `sample_id`.
4. Divisão treino/teste com 20% dos dados para teste.
5. Estratificação pela variável alvo para manter a distribuição das classes.
6. Normalização das variáveis numéricas com `StandardScaler`.
7. Codificação da variável categórica `region` com `OneHotEncoder`.
8. Treinamento de dois modelos supervisionados.
9. Avaliação por acurácia, precision, recall e f1-score.
10. Geração de matriz de confusão para cada modelo.
11. Seleção automática do melhor modelo por acurácia.
12. Geração de interpretabilidade com SHAP.
13. Demonstração funcional por aplicação Streamlit.

## 6. Modelos Testados

Foram comparadas duas técnicas preditivas:

| Modelo | Técnica | Justificativa |
|---|---|---|
| Random Forest Classifier | Ensemble de árvores | Bom desempenho em dados tabulares e robustez contra ruído |
| Gradient Boosting Classifier | Boosting sequencial | Modelo competitivo para padrões não lineares em classificação |

Os modelos treinados são salvos em:

```text
models/random_forest.joblib
models/gradient_boosting.joblib
models/best_model.joblib
```

## 7. Validação e Resultados

A base foi dividida em treino e teste com `test_size=0.2`, `random_state=42` e estratificação por classe. As métricas avaliadas foram acurácia, precision, recall e f1-score.

Resultados obtidos:

| Modelo | Acurácia |
|---|---:|
| Random Forest | 80,625% |
| Gradient Boosting | 80,3125% |

O modelo selecionado automaticamente como melhor foi:

```text
random_forest
```

O desempenho dos dois modelos ficou próximo, mas o Random Forest apresentou a maior acurácia no conjunto de teste. Os relatórios completos de classificação estão em:

```text
reports/metrics.json
```

As matrizes de confusão geradas estão em:

```text
reports/figures/confusion_matrix_random_forest.png
reports/figures/confusion_matrix_gradient_boosting.png
```

## 8. Interpretabilidade com SHAP

Foi utilizada a biblioteca SHAP para identificar as variáveis que mais influenciaram as previsões do melhor modelo. Como o problema é multiclasse, os valores SHAP foram agregados por importância média absoluta.

Principais variáveis:

| Posição | Variável | Importância SHAP |
|---:|---|---:|
| 1 | `temperature_anomaly_c` | 0.1471 |
| 2 | `sea_surface_temperature_c` | 0.0602 |
| 3 | `wave_height_m` | 0.0289 |
| 4 | `wind_speed_kmh` | 0.0221 |
| 5 | `pressure_hpa` | 0.0209 |

Interpretação técnica: a anomalia térmica foi a variável mais importante para a decisão do modelo. Isso é coerente com o problema, pois variações anormais na temperatura da superfície do mar indicam possível instabilidade climática e alterações ambientais. A temperatura da superfície, altura das ondas, vento e pressão atmosférica também aparecem com influência relevante, reforçando a coerência entre as regras de geração dos dados e o comportamento aprendido pelo modelo.

Arquivos gerados:

```text
reports/figures/shap_feature_importance.png
reports/shap_ranking.json
```

## 9. Deploy e Demonstração

A solução possui uma aplicação Streamlit implementada em:

```text
app.py
```

A interface permite que o usuário informe dados ambientais de uma região e receba:

- nível de risco previsto;
- confiança aproximada da predição;
- métricas do treinamento;
- ranking das variáveis mais importantes segundo SHAP.

Execução local:

```bash
streamlit run app.py
```

URL local:

```text
http://localhost:8501
```

Para a entrega final, falta apenas publicar a aplicação e preencher a URL pública no README e neste relatório.

## 10. Reprodutibilidade

Comandos de execução:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train_models.py
python src/shap_analysis.py
streamlit run app.py
```

Arquivos principais:

```text
app.py
requirements.txt
data/oceanmind_dataset.csv
src/generate_dataset.py
src/train_models.py
src/shap_analysis.py
src/predict.py
reports/metrics.json
reports/shap_ranking.json
```

## 11. Atendimento aos Critérios de Avaliação

| Critério | Evidência no projeto |
|---|---|
| Definição do problema e qualidade dos dados | Contexto definido, dataset sintético com 1.600 linhas e 15 colunas |
| Pré-processamento e engenharia de atributos | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder` e seleção de variáveis |
| Aplicação e comparação de modelos | Random Forest e Gradient Boosting treinados e comparados |
| Validação e análise de métricas | `metrics.json`, classification report e matrizes de confusão |
| Interpretabilidade com SHAP | `shap_analysis.py`, `shap_ranking.json` e gráfico de importância |
| Deploy da aplicação | Aplicação Streamlit pronta para publicação |
| Organização do código e README no GitHub | Estrutura modular com README completo |

## 12. Conclusão

O projeto OceanMind atende aos requisitos da disciplina GAIE ao apresentar uma solução de Machine Learning conectada à Indústria Espacial, com dataset sintético coerente, pipeline completo, dois modelos comparados, métricas de avaliação, seleção do melhor modelo, interpretabilidade com SHAP e aplicação Streamlit para demonstração.

A entrega está pronta para o repositório GitHub. A única pendência operacional é inserir a URL pública da aplicação após o deploy.

