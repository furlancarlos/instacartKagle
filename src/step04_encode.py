# src/step04_encode.py
# TransactionEncoder (matriz booleana)
# Rode com:  python -m src.step04_encode
# ============================================================

import sys
import gc
import os
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from src.config import CACHE_DIR, N_SAMPLE_ORDERS
from src.utils import timer, get_logger

log = get_logger('step04', 'step04_encode.log')


# ------------------------------------------------------------
# Medição de RAM (opcional)
# ------------------------------------------------------------
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
    else:
        log.info(f"[RAM] {label}: (psutil não instalado)")


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def load_transactions() -> list[tuple]:
    path = CACHE_DIR / 'transactions.parquet'
    log.info(f"Carregando transações de {path.name} ...")
    with timer("load transactions"):
        df = pd.read_parquet(path)
        # garante que é tupla
        df['basket'] = df['basket'].apply(tuple)
        baskets = df['basket'].tolist()
    log.info(f"  cestas carregadas: {len(baskets):,}")
    return baskets


def encode(baskets: list[tuple]) -> pd.DataFrame:
    log.info("Aplicando TransactionEncoder ...")
    log_ram("antes do encoder")
    with timer("TransactionEncoder.fit"):
        te = TransactionEncoder()
        te_array = te.fit(baskets).transform(baskets)
    log_ram("após transform")

    log.info(f"  matriz bruta: shape={te_array.shape} dtype={te_array.dtype}")
    n_cells = te_array.shape[0] * te_array.shape[1]
    log.info(f"  células: {n_cells:,}  ({n_cells / (1024**3):.2f} Gi células)")
    log.info(f"  memória teórica da matriz: "
             f"{n_cells / (1024**2):,.1f} MB (bool = 1 byte)")

    log.info("Convertendo para DataFrame ...")
    with timer("to DataFrame"):
        df = pd.DataFrame(te_array, columns=te.columns_)
        del te_array
        gc.collect()
    log_ram("após DataFrame")

    return df


def validate(df: pd.DataFrame) -> None:
    log.info("-" * 60)
    log.info("Sanity check:")
    log.info(f"  shape         : {df.shape}")
    log.info(f"  dtype         : {df.dtypes.unique().tolist()}")
    log.info(f"  memória (df)  : {df.memory_usage(deep=True).sum() / (1024**2):,.1f} MB")

    total_true = int(df.values.sum())
    log.info(f"  total de True : {total_true:,}")

    # Estatísticas por item
    item_freq = df.sum(axis=0).sort_values(ascending=False)
    log.info(f"  item mais frequente: '{item_freq.index[0]}' "
             f"({int(item_freq.iloc[0]):,} cestas)")
    log.info(f"  item menos frequente: '{item_freq.index[-1]}' "
             f"({int(item_freq.iloc[-1]):,} cestas)")
    log.info(f"  mediana de frequência: {item_freq.median():.0f} cestas")

    # Estatísticas por cesta
    basket_sizes = df.sum(axis=1)
    log.info(f"  cesta mínima  : {int(basket_sizes.min())}")
    log.info(f"  cesta mediana : {int(basket_sizes.median())}")
    log.info(f"  cesta máxima  : {int(basket_sizes.max())}")


def save(df: pd.DataFrame) -> None:
    out_path = CACHE_DIR / f'encoded_{N_SAMPLE_ORDERS // 1000}k.parquet'
    log.info(f"Salvando matriz em {out_path.name} ...")
    with timer("save parquet"):
        df.to_parquet(out_path, compression='snappy', index=False)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    log.info(f"  salvo: {out_path}  ({size_mb:,.1f} MB)")
    return out_path


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 04 — TransactionEncoder (matriz booleana)")
    log.info("=" * 60)
    log.info(f"Parâmetro: N_SAMPLE_ORDERS={N_SAMPLE_ORDERS:,}")
    log_ram("início do script")

    baskets = load_transactions()

    df = encode(baskets)
    del baskets
    gc.collect()

    validate(df)
    save(df)

    log_ram("fim do script")
    log.info("STEP 04 concluído com sucesso.")


if __name__ == '__main__':
    main()
