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

## Links de Entrega

| Item | Link |
|---|---|
| Repositório GitHub | https://github.com/pedrobicas/oceanmind-gaie-ml |
| Aplicação publicada |  |
| Execução local | http://localhost:8501 após rodar `streamlit run app.py` |

## Contexto do Problema

O OceanMind é uma solução de inteligência oceânica que utiliza dados espaciais, climáticos e ambientais para apoiar o monitoramento de riscos marítimos e costeiros. O projeto se conecta à Indústria Espacial ao simular o uso de dados derivados de observação terrestre e sensoriamento remoto para apoiar decisões em regiões oceânicas.

O problema tratado nesta entrega é a **classificação do nível de risco oceânico/climático** de uma região monitorada. A previsão pode apoiar órgãos ambientais, cidades costeiras, pesquisadores e operações marítimas na identificação de anomalias térmicas, condições meteorológicas severas e risco ambiental elevado.

## Objetivo do Modelo

Criar um pipeline completo de Machine Learning para prever a variável:

```text
ocean_risk_level
```

Classes previstas:

```text
baixo
moderado
alto
critico
```

## Fonte dos Dados

Foi utilizado um **dataset sintético** gerado por regras programadas em Python, coerentes com o contexto oceânico/climático. A geração está em:

```text
src/generate_dataset.py
```

O dataset final está em:

```text
data/oceanmind_dataset.csv
```

Características da base:

| Item | Valor |
|---|---:|
| Registros | 1.600 |
| Colunas | 15 |
| Variáveis preditoras usadas no modelo | 13 |
| Variável alvo | `ocean_risk_level` |
| Classes | `baixo`, `moderado`, `alto`, `critico` |

Distribuição das classes:

| Classe | Registros |
|---|---:|
| moderado | 431 |
| baixo | 394 |
| critico | 390 |
| alto | 385 |

Principais variáveis:

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

## Metodologia

O pipeline utiliza um fluxo supervisionado de classificação:

1. Carregamento do dataset CSV.
2. Remoção de colunas não preditivas (`sample_id`) e separação da variável alvo.
3. Divisão treino/teste com 20% para teste e estratificação por classe.
4. Pré-processamento com `ColumnTransformer`.
5. Normalização das variáveis numéricas com `StandardScaler`.
6. Codificação da variável categórica `region` com `OneHotEncoder`.
7. Treinamento de dois modelos.
8. Avaliação com acurácia, precision, recall, f1-score e matriz de confusão.
9. Seleção automática do melhor modelo por acurácia.
10. Geração de interpretabilidade com SHAP.
11. Demonstração funcional em Streamlit.

## Modelos Testados

Foram aplicadas duas técnicas diferentes de classificação:

| Modelo | Arquivo gerado |
|---|---|
| Random Forest Classifier | `models/random_forest.joblib` |
| Gradient Boosting Classifier | `models/gradient_boosting.joblib` |

O melhor modelo é salvo em:

```text
models/best_model.joblib
```

## Resultados Obtidos

Treinamento validado localmente em 29/05/2026.

| Modelo | Acurácia |
|---|---:|
| Random Forest | 80,625% |
| Gradient Boosting | 80,3125% |

Melhor modelo selecionado:

```text
random_forest
```

Arquivos gerados:

```text
models/random_forest.joblib
models/gradient_boosting.joblib
models/best_model.joblib
reports/metrics.json
reports/figures/confusion_matrix_random_forest.png
reports/figures/confusion_matrix_gradient_boosting.png
reports/figures/shap_feature_importance.png
reports/shap_ranking.json
```

## Interpretação com SHAP

A análise SHAP foi executada sobre o melhor modelo treinado e agregada por importância média absoluta. As variáveis mais influentes foram:

| Posição | Variável | Importância SHAP |
|---:|---|---:|
| 1 | `temperature_anomaly_c` | 0.1471 |
| 2 | `sea_surface_temperature_c` | 0.0602 |
| 3 | `wave_height_m` | 0.0289 |
| 4 | `wind_speed_kmh` | 0.0221 |
| 5 | `pressure_hpa` | 0.0209 |

Interpretação técnica: o modelo considera a **anomalia térmica** como o fator mais relevante para classificar o risco oceânico. Isso é coerente com o problema, pois desvios anormais de temperatura da superfície do mar tendem a indicar aquecimento local, instabilidade climática e alterações ambientais. Temperatura da superfície, altura das ondas, vento e pressão atmosférica também aparecem entre os fatores mais importantes, reforçando a consistência das regras usadas na geração do dataset.

## Aplicação Streamlit

A aplicação `app.py` permite informar dados ambientais de uma região e receber:

- classe prevista de risco oceânico;
- confiança aproximada do modelo;
- métricas de treinamento;
- ranking de variáveis SHAP.

Para publicar, use Streamlit Community Cloud, Azure App Service, Render ou Hugging Face Spaces. Depois da publicação, substitua o campo **Aplicação publicada** na seção de links.

## Como Executar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Gere novamente o dataset, se necessário:

```bash
python src/generate_dataset.py
```

Treine os modelos:

```bash
python src/train_models.py
```

Gere a análise SHAP:

```bash
python src/shap_analysis.py
```

Execute a aplicação:

```bash
streamlit run app.py
```

Abra no navegador:

```text
http://localhost:8501
```

## Relatório Técnico

O relatório final da disciplina está em:

```text
reports/relatorio_gaie.md
```

## Estrutura do Projeto

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   └── oceanmind_dataset.csv
├── models/
│   ├── best_model.joblib
│   ├── gradient_boosting.joblib
│   └── random_forest.joblib
├── notebooks/
│   └── oceanmind_ml_pipeline.ipynb
├── reports/
│   ├── metrics.json
│   ├── relatorio_gaie.md
│   ├── shap_ranking.json
│   └── figures/
│       ├── confusion_matrix_gradient_boosting.png
│       ├── confusion_matrix_random_forest.png
│       └── shap_feature_importance.png
└── src/
    ├── generate_dataset.py
    ├── predict.py
    ├── shap_analysis.py
    └── train_models.py
```

## Checklist dos Requisitos GAIE

| Requisito do enunciado | Status |
|---|---|
| Problema real relacionado à Economia/Indústria Espacial | Atendido |
| Dados via API ou sintéticos com no mínimo 1.000 linhas e 10 colunas | Atendido: 1.600 linhas e 15 colunas |
| Pelo menos duas técnicas preditivas | Atendido: Random Forest e Gradient Boosting |
| Pipeline com pré-processamento | Atendido |
| Engenharia/seleção de atributos | Atendido por seleção de variáveis e transformação numérica/categórica |
| Treinamento dos modelos | Atendido |
| Validação e comparação de desempenho | Atendido |
| Escolha do melhor modelo | Atendido |
| Deploy com Streamlit ou similar | Atendido localmente; falta apenas inserir a URL pública |
| Interpretabilidade com SHAP | Atendido |
| README detalhado | Atendido |
| Link do GitHub | Atendido |
| Link da aplicação em funcionamento | Pendente até publicação do deploy |

