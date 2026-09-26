# Mineração de Dados — Regras de Associação no Dataset Instacart

Trabalho prático da disciplina de **Mineração de Dados** aplicando o algoritmo
**FP-Growth** (Regras de Associação) ao dataset **Instacart Market Basket Analysis**.

## 📋 Sobre o Projeto

O objetivo é identificar padrões de compra conjunta em três níveis de granularidade
(produto, aisle e department) para gerar insights de negócio acionáveis:
recomendação personalizada, layout de loja e promoções cruzadas.

**Dataset:** [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)
— ~3,4 milhões de pedidos, 32,4 milhões de itens, 49.688 produtos distintos.

## 🧠 Metodologia

- **Amostra:** 400.000 pedidos aleatórios (seed=42)
- **Filtro:** produtos com frequência ≥ 100 na amostra (5.992 produtos)
- **Codificação:** `TransactionEncoder` (mlxtend)
- **Algoritmo:** FP-Growth
- **Regras:** `association_rules` com filtros de suporte, confiança e lift

## 📊 Principais Resultados

| Nível | Itemsets len≥2 | Regras finais |
|---|---|---|
| Produto | 2.495 | 144 |
| Aisle | 627 | 558 |
| Department | 228 | 369 |

**Top regras (por lift):**

- Cebola orgânica → Alho orgânico (lift = 5,23)
- Limão → Coentro orgânico (lift = 5,21)
- Frutas + Queijos + Iogurte → Leite + Vegetais (lift = 2,69)

## ⚡ Comparação Apriori vs FP-Growth

| Cestas | FP-Growth | Apriori | Speedup |
|---|---|---|---|
| 5.000 | 0,21 s | 0,46 s | 2,20x |
| 40.000 | 3,18 s | 3,69 s | 1,16x |
| 80.000 | 7,56 s | 12,22 s | 1,62x |
| **160.000** | **14,71 s** | **242,04 s** | **16,46x** |

> O FP-Growth é assintoticamente superior em bases grandes. O Apriori
> seria inviável no dataset completo (368k cestas).

## 🗂️ Estrutura do Projeto

P1/
├── database/ # Dataset (não versionado — baixar do Kaggle)
├── docs/ # PDF do enunciado
├── results/
│ ├── cache/ # Parquets intermediários (não versionado)
│ ├── figures/ # Gráficos (PNG)
│ ├── logs/ # Logs de execução
│ ├── report/ # Relatório final (Markdown + HTML)
│ ├── slides/ # Apresentação (PPTX)
│ └── tables/ # Tabelas (CSV)
├── src/ # Código-fonte
│ ├── config.py
│ ├── utils.py
│ ├── step01_setup.py
│ ├── step02_load_eda.py
│ ├── step03_transactions.py
│ ├── step04_encode.py
│ ├── step05b_fpgrowth_produto.py
│ ├── step05c_build_aisle_department.py
│ ├── step05d_fpgrowth_macro.py
│ ├── step06_rules.py
│ ├── step07_insights.py
│ ├── step08_report.py
│ ├── step09_slides.py
│ ├── step10_apriori_vs_fpgrowth.py
│ └── exploratory/
│ └── step05_fpgrowth.py
├── .gitignore
├── requirements.txt
└── README.md


## 🚀 Como Reproduzir

### 1. Clonar o repositório

git clone https://github.com/<seu-usuario>/<seu-repo>.git
cd <seu-repo>/P1

### 2. Baixar o dataset

Baixe os 6 arquivos CSV do Kaggle
e coloque-os em database/:
text

database/
├── aisles.csv
├── departments.csv
├── order_products__prior.csv
├── order_products__train.csv
├── orders.csv
└── products.csv

### 3. Criar ambiente virtual

Windows (CMD):

CMD

python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

CMD

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 4. Executar o pipeline (em ordem)

CMD

python -m src.step01_setup
python -m src.step02_load_eda
python -m src.step03_transactions
python -m src.step04_encode
python -m src.step05b_fpgrowth_produto
python -m src.step05c_build_aisle_department
python -m src.step05d_fpgrowth_macro
python -m src.step06_rules
python -m src.step07_insights
python -m src.step08_report
python -m src.step09_slides
python -m src.step10_apriori_vs_fpgrowth

Tempo total: ~15-20 min (primeira execução; step02 e step10 são os mais longos).

🛠️ Tecnologias

    Python 3.13+

    pandas 3.0.6 — manipulação de dados

    NumPy 2.5.3 — operações vetorizadas

    mlxtend 0.25.0 — FP-Growth, Apriori, association_rules

    pyarrow 25.0.1 — leitura/escrita de parquet

    matplotlib + seaborn — visualizações

    python-pptx — geração automática de slides

    psutil — medição de RAM

📄 Licença

Projeto acadêmico — uso livre para fins educacionais.

👥 Autores

    Carlos Alberto Furlan

    Gustavo Henrique Macedo Pena

Disciplina: Mineração de Dados

text


---

## 5. Sequência de comandos

### Passo 5.1 — Inicializar o repositório local (se ainda não fez)

Dentro de `P1/`:

```cmd
git init
