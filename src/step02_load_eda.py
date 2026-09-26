# src/step02_load_eda.py
# Carregamento + Exploratory Data Analysis (EDA) inicial
# Rode com:  python -m src.step02_load_eda
# ============================================================

import os
import sys
import gc
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FILES, FIG_DIR, CACHE_DIR, CHUNK_SIZE
from src.utils import timer, get_logger, human_size

log = get_logger('step02', 'step02_load_eda.log')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_small_tables() -> dict:
    """Carrega as tabelas pequenas do dataset Instacart."""
    log.info("Carregando tabelas pequenas...")

    dfs = {}
    with timer("aisles"):
        dfs['aisles'] = pd.read_csv(FILES['aisles'])
        log.info(f"  aisles: {dfs['aisles'].shape}")

    with timer("departments"):
        dfs['departments'] = pd.read_csv(FILES['departments'])
        log.info(f"  departments: {dfs['departments'].shape}")

    with timer("products"):
        dfs['products'] = pd.read_csv(FILES['products'])
        log.info(f"  products: {dfs['products'].shape}")

    with timer("orders"):
        # order_id e user_id como int32 economiza memória
        dtypes = {
            'order_id': 'int32',
            'user_id': 'int32',
            'eval_set': 'category',
            'order_number': 'int16',
            'order_dow': 'int8',
            'order_hour_of_day': 'int8',
            'days_since_prior_order': 'float32',
        }
        dfs['orders'] = pd.read_csv(FILES['orders'], dtype=dtypes)
        log.info(f"  orders: {dfs['orders'].shape}")

    return dfs


def load_prior_chunked(save_parquet: bool = True) -> pd.DataFrame:
    """
    Lê order_products__prior.csv em chunks e agrega
    order_id -> lista de product_ids (ordenados por add_to_cart_order).
    Salva em parquet para reuso nos próximos steps.
    """
    cache_path = CACHE_DIR / 'prior_by_order.parquet'

    if cache_path.exists():
        log.info(f"Cache encontrado: {cache_path.name}. Carregando parquet...")
        with timer("load prior (cache)"):
            df = pd.read_parquet(cache_path)
            # converter lista -> tuple para ficar mais leve na memória
            df['products'] = df['products'].apply(tuple)
            log.info(f"  prior (cache): {df.shape}")
        return df

    log.info(f"Lendo {FILES['prior'].name} em chunks de {CHUNK_SIZE:,} linhas...")
    t0 = __import__('time').time()

    # Vamos acumular num dict: order_id -> list[int]
    baskets: dict[int, list[int]] = {}
    total_rows = 0
    chunk_idx = 0

    dtypes = {
        'order_id': 'int32',
        'product_id': 'int32',
        'add_to_cart_order': 'int16',
        'reordered': 'int8',
    }

    reader = pd.read_csv(
        FILES['prior'],
        usecols=['order_id', 'product_id', 'add_to_cart_order'],
        dtype=dtypes,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        chunk_idx += 1
        total_rows += len(chunk)

        # ordena por (order_id, add_to_cart_order) para preservar a ordem do carrinho
        chunk = chunk.sort_values(['order_id', 'add_to_cart_order'],
                                  kind='mergesort')

        # agrupa em listas
        for oid, grp in chunk.groupby('order_id', sort=False)['product_id']:
            lst = baskets.setdefault(int(oid), [])
            lst.extend(grp.tolist())

        log.info(f"  chunk {chunk_idx:>2} | rows acumuladas: {total_rows:>12,} "
                 f"| pedidos acumulados: {len(baskets):>10,}")

        del chunk
        gc.collect()

    dt = __import__('time').time() - t0
    log.info(f"Leitura total concluída em {dt:.1f}s "
             f"| {total_rows:,} linhas | {len(baskets):,} pedidos")

    # Monta DataFrame final
    df = pd.DataFrame(
        {'order_id': list(baskets.keys()),
         'products': [tuple(v) for v in baskets.values()]}
    )
    del baskets
    gc.collect()

    log.info(f"DataFrame agregado: {df.shape}")

    if save_parquet:
        log.info(f"Salvando cache em {cache_path} ...")
        df.to_parquet(cache_path, index=False)
        log.info(f"  parquet salvo: {human_size(cache_path.stat().st_size)}")

    return df


# ------------------------------------------------------------
# EDA
# ------------------------------------------------------------
def eda_summary(dfs: dict, prior_df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("EDA — Visão geral")
    log.info("=" * 60)

    for name, df in dfs.items():
        log.info(f"{name:12s} shape={df.shape}  "
                 f"nulos={int(df.isna().sum().sum())}  "
                 f"duplicados={int(df.duplicated().sum())}")

    log.info(f"{'prior':12s} shape={prior_df.shape}  "
             f"(1 linha por pedido)")

    # Estatísticas do número de produtos por pedido
    n_items = prior_df['products'].apply(len)
    log.info("-" * 60)
    log.info("Nº de produtos por pedido:")
    log.info(f"  min     : {n_items.min()}")
    log.info(f"  mediana : {n_items.median():.0f}")
    log.info(f"  média   : {n_items.mean():.2f}")
    log.info(f"  max     : {n_items.max()}")
    log.info(f"  p95     : {n_items.quantile(0.95):.0f}")
    log.info(f"  p99     : {n_items.quantile(0.99):.0f}")

    # Total de itens
    log.info(f"Total de itens em pedidos (prior): {int(n_items.sum()):,}")

    return n_items


def make_plots(dfs: dict, prior_df: pd.DataFrame, n_items: pd.Series) -> None:
    log.info("Gerando gráficos...")

    products    = dfs['products']
    aisles      = dfs['aisles']
    departments = dfs['departments']

    # --------------------------------------------------------
    # Tabela "produto enriquecido" com aisle e department
    # products.csv tem: product_id, product_name, aisle_id, department_id
    # aisles.csv    tem: aisle_id, aisle
    # departments.csv tem: department_id, department
    # --------------------------------------------------------
    prod_full = (
        products
        .merge(aisles[['aisle_id', 'aisle']], on='aisle_id', how='left')
        .merge(departments[['department_id', 'department']],
               on='department_id', how='left')
    )
    log.info(f"prod_full shape: {prod_full.shape}")
    log.info(f"prod_full nulos em 'department': "
             f"{int(prod_full['department'].isna().sum())}")

    # --------------------------------------------------------
    # Contagem global de itens (uma vez só, reaproveitada)
    # --------------------------------------------------------
    all_items = pd.DataFrame({
        'product_id': np.concatenate(prior_df['products'].values)
    })
    all_items = all_items.merge(
        prod_full[['product_id', 'product_name', 'aisle', 'department']],
        on='product_id', how='left'
    )
    log.info(f"all_items shape: {all_items.shape}")

    # --------------------------------------------------------
    # 1) Top 20 produtos
    # --------------------------------------------------------
    top = (
        all_items['product_name'].value_counts().head(20)
        .reset_index()
    )
    top.columns = ['product_name', 'n']

    plt.figure(figsize=(12, 7))
    sns.barplot(data=top, y='product_name', x='n',
                hue='product_name', palette='viridis', legend=False)
    plt.title('Top 20 produtos mais comprados (pedidos prior)')
    plt.xlabel('Nº de compras')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top20_products.png', dpi=120, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------
    # 2) Top 15 aisles
    # --------------------------------------------------------
    top_aisle = (
        all_items['aisle'].value_counts().head(15)
        .reset_index()
    )
    top_aisle.columns = ['aisle', 'n']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_aisle, y='aisle', x='n',
                hue='aisle', palette='magma', legend=False)
    plt.title('Top 15 corredores (aisles) mais frequentes')
    plt.xlabel('Nº de itens vendidos')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top15_aisles.png', dpi=120, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------
    # 3) Top 10 departamentos
    # --------------------------------------------------------
    top_dept = (
        all_items['department'].value_counts().head(10)
        .reset_index()
    )
    top_dept.columns = ['department', 'n']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_dept, y='department', x='n',
                hue='department', palette='crest', legend=False)
    plt.title('Top 10 departamentos mais frequentes')
    plt.xlabel('Nº de itens vendidos')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top10_departments.png', dpi=120, bbox_inches='tight')
    plt.close()

    # --------------------------------------------------------
    # 4) Distribuição nº de produtos por pedido
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    sns.histplot(n_items, bins=80, color='steelblue')
    plt.title('Distribuição do nº de produtos por pedido')
    plt.xlabel('Nº de produtos no pedido')
    plt.ylabel('Nº de pedidos')
    plt.xlim(0, 80)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'dist_items_per_order.png',
                dpi=120, bbox_inches='tight')
    plt.close()

    del all_items
    gc.collect()

    log.info(f"Gráficos salvos em: {FIG_DIR}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 02 — Carregamento + EDA inicial")
    log.info("=" * 60)

    dfs = load_small_tables()
    prior_df = load_prior_chunked(save_parquet=True)

    n_items = eda_summary(dfs, prior_df)
    make_plots(dfs, prior_df, n_items)

    # Salva algumas tabelas-resumo em CSV
    summary = pd.DataFrame({
        'order_id': prior_df['order_id'].values,
        'n_products': n_items.values,
    })
    summary.to_csv(CACHE_DIR.parent / 'tables' / 'orders_n_products.csv',
                   index=False)
    log.info("Resumo (order_id, n_products) salvo em results/tables/")

    log.info("STEP 02 concluído com sucesso.")


if __name__ == '__main__':
    main()
