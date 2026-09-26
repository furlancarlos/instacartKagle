# src/step03_transactions.py
# Amostragem + filtro + transações
# Rode com:  python -m src.step03_transactions
# ============================================================

import sys
import gc
from pathlib import Path
from collections import Counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    FILES, CACHE_DIR, N_SAMPLE_ORDERS, MIN_PRODUCT_FREQ, RANDOM_SEED
)
from src.utils import timer, get_logger

log = get_logger('step03', 'step03_transactions.log')


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_prior() -> pd.DataFrame:
    cache_path = CACHE_DIR / 'prior_by_order.parquet'
    log.info(f"Carregando prior de {cache_path.name} ...")
    with timer("load prior"):
        df = pd.read_parquet(cache_path)
        df['products'] = df['products'].apply(tuple)
    log.info(f"  prior: {df.shape}")
    return df


def load_product_names() -> dict[int, str]:
    log.info("Carregando products.csv ...")
    prod = pd.read_csv(FILES['products'], usecols=['product_id', 'product_name'])
    # limpa espaços e normaliza
    prod['product_name'] = (
        prod['product_name']
        .astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
    )
    # dicionário product_id -> product_name
    mapping = dict(zip(prod['product_id'].astype(int), prod['product_name']))
    log.info(f"  produtos mapeados: {len(mapping):,}")
    return mapping


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def sample_orders(prior: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(prior):
        log.info(f"Amostra solicitada ({n:,}) >= total ({len(prior):,}). "
                 f"Usando todos os pedidos.")
        return prior.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prior), size=n, replace=False)
    idx.sort()  # mantém ordem
    sample = prior.iloc[idx].reset_index(drop=True)
    log.info(f"Amostrados {len(sample):,} pedidos de {len(prior):,} "
             f"({100 * n / len(prior):.2f}%)")
    return sample


def count_product_freq(sample: pd.DataFrame) -> Counter:
    log.info("Contando frequência de product_id na amostra ...")
    with timer("count products"):
        counter = Counter()
        for lst in sample['products']:
            counter.update(lst)
    log.info(f"  produtos distintos na amostra: {len(counter):,}")
    return counter


def filter_baskets(sample: pd.DataFrame,
                   counter: Counter,
                   min_freq: int,
                   mapping: dict[int, str]) -> pd.DataFrame:
    log.info(f"Filtrando produtos com frequência >= {min_freq} ...")
    valid = {pid for pid, c in counter.items() if c >= min_freq}
    log.info(f"  produtos válidos: {len(valid):,} "
             f"({100 * len(valid) / len(counter):.2f}% do total)")

    # Verifica cobertura: qual % dos itens originais ficaram
    total_items = sum(counter.values())
    covered_items = sum(c for pid, c in counter.items() if pid in valid)
    log.info(f"  cobertura de itens: {covered_items:,}/{total_items:,} "
             f"({100 * covered_items / total_items:.2f}%)")

    log.info("Reconstruindo cestas apenas com produtos válidos ...")
    with timer("rebuild baskets"):
        new_baskets = []
        for lst in sample['products']:
            filt = [pid for pid in lst if pid in valid]
            if len(filt) >= 2:  # cesta precisa ter >= 2 produtos p/ gerar regras
                new_baskets.append(filt)

    log.info(f"  cestas com >= 2 produtos: {len(new_baskets):,} "
             f"(de {len(sample):,} originais)")

    # Converte para nomes
    log.info("Convertendo product_id -> product_name ...")
    with timer("map names"):
        named_baskets = [
            tuple(mapping[pid] for pid in lst if pid in mapping)
            for lst in new_baskets
        ]

    # Remove cestas que ficaram < 2 após mapeamento (produto sem nome)
    named_baskets = [b for b in named_baskets if len(b) >= 2]
    log.info(f"  cestas finais: {len(named_baskets):,}")

    df = pd.DataFrame({'basket': named_baskets})
    return df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 03 — Amostragem + filtro + construção de transações")
    log.info("=" * 60)
    log.info(f"Parâmetros: N_SAMPLE_ORDERS={N_SAMPLE_ORDERS:,} | "
             f"MIN_PRODUCT_FREQ={MIN_PRODUCT_FREQ} | "
             f"RANDOM_SEED={RANDOM_SEED}")

    # 1) Carrega prior
    prior = load_prior()

    # 2) Amostra
    sample = sample_orders(prior, N_SAMPLE_ORDERS, RANDOM_SEED)
    del prior
    gc.collect()

    # 3) Carrega nomes
    mapping = load_product_names()

    # 4) Conta frequência
    counter = count_product_freq(sample)

    # 5) Filtra e reconstrói cestas
    baskets_df = filter_baskets(sample, counter, MIN_PRODUCT_FREQ, mapping)

    # 6) Estatísticas finais
    sizes = baskets_df['basket'].apply(len)
    log.info("-" * 60)
    log.info("Estatísticas das cestas finais:")
    log.info(f"  nº cestas     : {len(baskets_df):,}")
    log.info(f"  tamanho min   : {sizes.min()}")
    log.info(f"  mediana       : {sizes.median():.0f}")
    log.info(f"  média         : {sizes.mean():.2f}")
    log.info(f"  max           : {sizes.max()}")
    log.info(f"  total de itens: {sizes.sum():,}")

    # 7) Salva
    out_path = CACHE_DIR / 'transactions.parquet'
    log.info(f"Salvando transações em {out_path.name} ...")
    with timer("save parquet"):
        baskets_df.to_parquet(out_path, index=False)
    log.info(f"  salvo: {out_path}")

    # Bônus: 10 exemplos de cestas (para o relatório)
    log.info("-" * 60)
    log.info("Exemplos de cestas (10 primeiras):")
    for i, row in baskets_df.head(10).iterrows():
        items = ', '.join(row['basket'])
        log.info(f"  [{i}] ({len(row['basket'])}) {items[:180]}"
                 f"{'...' if len(items) > 180 else ''}")

    log.info("STEP 03 concluído com sucesso.")


if __name__ == '__main__':
    main()
