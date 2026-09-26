# src/step01_setup.py
# Verificação do ambiente e arquivos
# Rode com:  python -m src.step01_setup
# ============================================================

import os
import sys
from pathlib import Path

# Permite rodar direto: python src\step01_setup.py
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.config import BASE_DIR, DATA_DIR, RESULTS, FILES
from src.utils import human_size, get_logger

log = get_logger('step01', 'step01_setup.log')

EXPECTED = [
    'orders.csv',
    'order_products__prior.csv',
    'order_products__train.csv',
    'products.csv',
    'aisles.csv',
    'departments.csv',
]


def main() -> None:
    log.info("=" * 60)
    log.info("STEP 01 — Verificação do ambiente")
    log.info("=" * 60)

    log.info(f"BASE_DIR : {BASE_DIR}")
    log.info(f"DATA_DIR : {DATA_DIR}")
    log.info(f"RESULTS  : {RESULTS}")

    log.info("-" * 60)
    log.info("Arquivos em database/:")
    encontrados = {}
    for f in sorted(os.listdir(DATA_DIR)):
        p = DATA_DIR / f
        if p.is_file():
            size = human_size(p.stat().st_size)
            encontrados[f] = p
            log.info(f"  {size:>10}  {f}")

    log.info("-" * 60)
    log.info("Checagem dos arquivos esperados:")
    faltando = []
    for f in EXPECTED:
        ok = f in encontrados
        log.info(f"  {'✓' if ok else '✗'} {f}")
        if not ok:
            faltando.append(f)

    if faltando:
        log.error(f"Arquivos faltando: {faltando}")
        sys.exit(1)

    log.info("-" * 60)
    log.info(f"Pandas  : {pd.__version__}")
    log.info(f"NumPy   : {np.__version__}")

    try:
        import mlxtend
        log.info(f"mlxtend : {mlxtend.__version__}")
    except Exception as e:
        log.warning(f"mlxtend não disponível: {e}")

    try:
        import pyarrow
        log.info(f"pyarrow : {pyarrow.__version__}")
    except Exception as e:
        log.warning(f"pyarrow não disponível: {e}")

    log.info("STEP 01 concluído com sucesso.")


if __name__ == '__main__':
    main()
