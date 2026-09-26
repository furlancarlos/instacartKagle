# src/step05c_build_aisle_department.py
# Constrói transações em nível de AISLE e DEPARTMENT
# Rode com:  python -m src.step05c_build_aisle_department
# ============================================================

import sys
import os
import gc
from pathlib import Path
from collections import Counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from src.config import (
    FILES, CACHE_DIR, N_SAMPLE_ORDERS
)
from src.utils import timer, get_logger

log = get_logger('step05c', 'step05c_build_aisle_department.log')


try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def ram_mb() -> float:
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def log_ram(label: str) -> None:
    if _HAS_PSUTIL:
        log.info(f"[RAM] {label}: {ram_mb():,.1f} MB")


# ------------------------------------------------------------
# Carregamento
# ------------------------------------------------------------
def load_transactions() -> pd.DataFrame:
    path = CACHE_DIR / 'transactions.parquet'
    log.info(f"Carregando transações de {path.name} ...")
    with timer("load transactions"):
        df = pd.read_parquet(path)
        df['basket'] = df['basket'].apply(tuple)
    log.info(f"  cestas: {len(df):,}")
    return df


def load_product_maps() -> tuple[dict, dict]:
    """Retorna (product_name -> aisle, product_name -> department)."""
    log.info("Carregando products.csv, aisles.csv, departments.csv ...")

    products = pd.read_csv(FILES['products'])
    aisles = pd.read_csv(FILES['aisles'])
    departments = pd.read_csv(FILES['departments'])

    # limpa nomes (mesma limpeza do step03)
    products['product_name'] = (
        products['product_name'].astype(str).str.strip()
                                    .str.replace(r'\s+', ' ', regex=True)
    )

    prod_full = (
        products
        .merge(aisles[['aisle_id', 'aisle']], on='aisle_id', how='left')
        .merge(departments[['department_id', 'department']],
               on='department_id', how='left')
    )

    map_aisle = dict(zip(prod_full['product_name'], prod_full['aisle']))
    map_dept  = dict(zip(prod_full['product_name'], prod_full['department']))

    log.info(f"  produtos mapeados: {len(map_aisle):,}")
    log.info(f"  aisles distintos:  {prod_full['aisle'].nunique()}")
    log.info(f"  departments distintos: {prod_full['department'].nunique()}")

    return map_aisle, map_dept


# ------------------------------------------------------------
# Construção de cestas macro
# ------------------------------------------------------------
def build_baskets(baskets_prod: list[tuple],
                  mapping: dict,
                  label: str) -> list[tuple]:
    """Converte cestas de produtos -> cestas de categorias (sem repetição)."""
    log.info(f"Convertendo cestas para nível de {label} ...")
    with timer(f"build {label}"):
        out = []
        unmapped = 0
        for basket in baskets_prod:
            cats = set()
            for prod in basket:
                cat = mapping.get(prod)
                if cat is None:
                    unmapped += 1
                    continue
                cats.add(cat)
            if len(cats) >= 2:      # precisa >=2 categorias para gerar regras
                out.append(tuple(sorted(cats)))
    log.info(f"  cestas válidas: {len(out):,}")
    log.info(f"  produtos sem mapeamento: {unmapped:,} (ignorados)")
    return out


def encode_baskets(baskets: list[tuple], label: str) -> pd.DataFrame:
    log.info(f"Aplicando TransactionEncoder [{label}] ...")
    log_ram(f"antes [{label}]")
    with timer(f"encode {label}"):
        te = TransactionEncoder()
        arr = te.fit(baskets).transform(baskets)
        df = pd.DataFrame(arr, columns=te.columns_)
        del arr, te
        gc.collect()
    log.info(f"  shape: {df.shape}  "
             f"| memória: {df.memory_usage(deep=True).sum() / (1024**2):,.1f} MB")
    log_ram(f"depois [{label}]")
    return df


def save_encoded(df: pd.DataFrame, label: str) -> Path:
    out = CACHE_DIR / f'encoded_{label}_{N_SAMPLE_ORDERS // 1000}k.parquet'
    log.info(f"Salvando em {out.name} ...")
    with timer(f"save {label}"):
        df.to_parquet(out, compression='snappy', index=False)
    log.info(f"  salvo: {out}  ({out.stat().st_size / (1024**2):,.1f} MB)")
    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 05c — Construir transações AISLE e DEPARTMENT")
    log.info("=" * 60)
    log.info(f"N_SAMPLE_ORDERS={N_SAMPLE_ORDERS:,}")
    log_ram("início")

    # 1) Carrega transações de produto
    df_prod = load_transactions()
    baskets_prod = df_prod['basket'].tolist()
    del df_prod
    gc.collect()

    # 2) Carrega mapeamentos
    map_aisle, map_dept = load_product_maps()

    # 3) Constrói cestas de aisle e department
    baskets_aisle = build_baskets(baskets_prod, map_aisle, 'AISLE')
    baskets_dept  = build_baskets(baskets_prod, map_dept,  'DEPARTMENT')
    del baskets_prod
    gc.collect()

    # 4) Codifica
    enc_aisle = encode_baskets(baskets_aisle, 'aisle')
    save_encoded(enc_aisle, 'aisle')

    enc_dept = encode_baskets(baskets_dept, 'department')
    save_encoded(enc_dept, 'department')

    # 5) Resumo
    log.info("=" * 60)
    log.info("Resumo:")
    log.info(f"  AISLE      : {enc_aisle.shape}  "
             f"(média de {enc_aisle.sum(axis=1).mean():.2f} aisles/cesta)")
    log.info(f"  DEPARTMENT : {enc_dept.shape}  "
             f"(média de {enc_dept.sum(axis=1).mean():.2f} depts/cesta)")

    log_ram("fim")
    log.info("STEP 05c concluído.")


if __name__ == '__main__':
    main()
