# Architecture / Arquitetura

## English

### System Architecture

```mermaid
graph TD
    A[Data Source] --> B[data_loader.py]
    B --> C[Synthetic HR Dataset]
    C --> D[eda.py]
    C --> E[attrition_model.py]
    D --> F[EDA Report]
    D --> G[Salary Equity Analysis]
    D --> H[Outlier Detection]
    E --> I[Trained Model .pkl]
    E --> J[Feature Importances]
    F --> K[pipeline.py]
    G --> K
    I --> K
    K --> L[Final JSON Report]
    K --> M[Artifacts on Disk]
```

### Pipeline Flow

```mermaid
sequenceDiagram
    participant CLI as CLI / Docker
    participant P as pipeline.py
    participant DL as data_loader.py
    participant EDA as eda.py
    participant M as attrition_model.py

    CLI->>P: run_pipeline(n=500)
    P->>DL: load_hr_data(n=500)
    DL-->>P: DataFrame (500 rows x 35 cols)
    P->>EDA: HRExploratoryAnalysis(df)
    EDA-->>P: EDA report + salary equity + outliers
    P->>M: AttritionModel.train(df)
    M-->>P: trained model
    P->>M: model.evaluate(df)
    M-->>P: ROC-AUC, classification report
    P-->>CLI: JSON results + saved artifacts
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data_loader.py` | Generate synthetic IBM HR-style data or load from CSV |
| `eda.py` | Statistical analysis, correlations, outlier detection, salary equity |
| `attrition_model.py` | RandomForest classifier with SMOTE for class imbalance |
| `pipeline.py` | Orchestrate all steps, save artifacts, generate reports |

### Directory Structure

```
pandas-data-analysis-hr/
├── .github/workflows/ci.yml   # CI/CD with GitHub Actions
├── docs/                       # Documentation
│   └── architecture.md
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & generation
│   ├── eda.py                  # Exploratory data analysis
│   ├── attrition_model.py      # ML model for attrition
│   └── pipeline.py             # End-to-end orchestrator
├── tests/                      # Unit tests (pytest)
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_eda.py
│   └── test_attrition_model.py
├── data/                       # Generated at runtime
│   ├── raw/
│   └── processed/
├── models/                     # Saved model artifacts
├── reports/                    # Generated reports
├── .env.example                # Environment variables template
├── Dockerfile                  # Multi-stage Docker build
├── Makefile                    # Dev commands (test, lint, run)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Bilingual documentation
```

---

## Portugues (PT-BR)

### Arquitetura do Sistema

O pipeline segue uma arquitetura modular com separacao clara de responsabilidades:

1. **Camada de Dados** (`data_loader.py`): Gera dados sinteticos no estilo IBM HR Attrition ou carrega CSVs existentes.
2. **Camada Analitica** (`eda.py`): Estatisticas descritivas, correlacoes, deteccao de outliers e analise de equidade salarial.
3. **Camada de Modelagem** (`attrition_model.py`): Classificador RandomForest com SMOTE para balanceamento de classes.
4. **Camada de Orquestracao** (`pipeline.py`): Executa todas as etapas sequencialmente, salva artefatos e gera relatorios.

### Fluxo de Dados

```mermaid
graph LR
    A[Dados Sinteticos] --> B[Analise Exploratoria]
    B --> C[Equidade Salarial]
    B --> D[Taxas de Attrition]
    B --> E[Deteccao de Outliers]
    A --> F[Modelo de Attrition]
    F --> G[Metricas ROC-AUC]
    F --> H[Feature Importances]
    C --> I[Relatorio Final]
    D --> I
    G --> I
    H --> I
```

### Tecnologias Utilizadas

| Tecnologia | Uso |
|-----------|-----|
| Python 3.10+ | Linguagem principal |
| Pandas | Manipulacao e analise de dados |
| Scikit-learn | Modelagem preditiva |
| imbalanced-learn | SMOTE para classes desbalanceadas |
| Pytest | Testes unitarios |
| Docker | Containerizacao |
| GitHub Actions | CI/CD |
