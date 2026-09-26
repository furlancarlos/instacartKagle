# src/config.py
#  Caminhos e constantes do projeto
# ============================================================

from pathlib import Path

# ------------------------------------------------------------
# Localizar a raiz do projeto (subindo até achar 'database/')
# ------------------------------------------------------------
def _find_project_root(start: Path, marker: str = 'database') -> Path:
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(
        f"Não encontrei a pasta '{marker}/' subindo a partir de {start}"
    )

BASE_DIR = _find_project_root(Path(__file__).parent)
DATA_DIR = BASE_DIR / 'database'
SRC_DIR  = BASE_DIR / 'src'
RESULTS  = BASE_DIR / 'results'

FIG_DIR  = RESULTS / 'figures'
TAB_DIR  = RESULTS / 'tables'
LOG_DIR  = RESULTS / 'logs'
CACHE_DIR = RESULTS / 'cache'   # parquet intermediário

for d in (FIG_DIR, TAB_DIR, LOG_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Arquivos do dataset
# ------------------------------------------------------------
FILES = {
    'orders':        DATA_DIR / 'orders.csv',
    'prior':         DATA_DIR / 'order_products__prior.csv',
    'train':         DATA_DIR / 'order_products__train.csv',
    'products':      DATA_DIR / 'products.csv',
    'aisles':        DATA_DIR / 'aisles.csv',
    'departments':   DATA_DIR / 'departments.csv',
}

# ------------------------------------------------------------
# Parâmetros do algoritmo (ajustaremos depois)
# ------------------------------------------------------------
# Amostragem
N_SAMPLE_ORDERS   = 400_000     # 100k -> 200k -> 400k conforme RAM
RANDOM_SEED       = 42

# Filtro de produtos raros
MIN_PRODUCT_FREQ  = 100         # produto precisa aparecer em >= N pedidos

# FP-Growth / Regras
MIN_SUPPORT       = 0.01        # será ajustado após ver a distribuição
MIN_CONFIDENCE    = 0.20
MIN_LIFT          = 1.0

# Performance
CHUNK_SIZE        = 2_000_000

# ------------------------------------------------------------
# Informações do trabalho
# ------------------------------------------------------------
INTEGRANTES = [
    'Carlos Alberto Furlan',
    'Gustavo Henrique Macedo Pena',
]
DISCIPLINA = 'Mineração de Dados'
