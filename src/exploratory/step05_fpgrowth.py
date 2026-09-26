# src/step05_fpgrowth.py
# FP-Growth: itemsets frequentes
# Rode com:  python -m src.step05_fpgrowth
# ============================================================

import sys
import os
import gc
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth

from src.config import CACHE_DIR, FIG_DIR, TAB_DIR, N_SAMPLE_ORDERS
from src.utils import timer, get_logger

log = get_logger('step05', 'step05_fpgrowth.log')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


# ------------------------------------------------------------
# RAM
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


# ------------------------------------------------------------
# Parâmetros
# ------------------------------------------------------------
SUPPORTS = [0.02, 0.01, 0.005]


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def load_encoded() -> pd.DataFrame:
    path = CACHE_DIR / f'encoded_{N_SAMPLE_ORDERS // 1000}k.parquet'
    log.info(f"Carregando matriz de {path.name} ...")
    with timer("load encoded"):
        df = pd.read_parquet(path)
    log.info(f"  shape: {df.shape}")
    log.info(f"  memória: {df.memory_usage(deep=True).sum() / (1024**2):,.1f} MB")
    return df


def run_fpgrowth(df: pd.DataFrame, min_support: float) -> tuple[pd.DataFrame, float]:
    log.info("-" * 60)
    log.info(f"FP-Growth com min_support = {min_support}")
    log_ram("antes do FP-Growth")

    t0 = time.time()
    itemsets = fpgrowth(
        df,
        min_support=min_support,
        use_colnames=True,
        max_len=None,          # sem limite de tamanho
        verbose=0,
    )
    dt = time.time() - t0

    log.info(f"  tempo: {dt:.2f}s")
    log.info(f"  itemsets encontrados: {len(itemsets):,}")
    log_ram("após FP-Growth")

    return itemsets, dt


def describe_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna de tamanho e retorna tabela-resumo por tamanho."""
    itemsets = itemsets.copy()
    itemsets['length'] = itemsets['itemsets'].apply(len)

    summary = (
        itemsets
        .groupby('length')
        .agg(
            n_itemsets=('support', 'size'),
            support_min=('support', 'min'),
            support_max=('support', 'max'),
            support_mean=('support', 'mean'),
        )
        .reset_index()
        .sort_values('length')
    )

    return itemsets, summary


def save_itemsets(itemsets: pd.DataFrame, min_support: float) -> Path:
    tag = f"support{min_support}".replace('.', '_')
    out = CACHE_DIR / f'itemsets_{tag}.parquet'
    # transforma frozenset em string ordenada (parquet não lida bem com frozenset)
    itemsets_out = itemsets.copy()
    itemsets_out['itemsets'] = itemsets_out['itemsets'].apply(
        lambda s: '||'.join(sorted(s))
    )
    itemsets_out.to_parquet(out, index=False)
    log.info(f"  salvo: {out.name}  "
             f"({out.stat().st_size / (1024**2):,.1f} MB)")
    return out


# ------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------
def plot_support_vs_itemsets(results: list[dict]) -> None:
    """Gráfico: min_support x nº de itemsets (e tempo)."""
    df = pd.DataFrame(results)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('min_support')
    ax1.set_ylabel('Nº de itemsets', color=color1)
    ax1.plot(df['min_support'], df['n_itemsets'], 'o-', color=color1, lw=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Tempo (s)', color=color2)
    ax2.plot(df['min_support'], df['time_s'], 's--', color=color2, lw=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('FP-Growth: impacto do min_support\n'
              '(nº de itemsets e tempo de execução)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fpgrowth_support_vs_itemsets.png',
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_length_distribution(all_summaries: dict) -> None:
    """Distribuição do tamanho dos itemsets para cada min_support."""
    plt.figure(figsize=(12, 6))
    for ms, summary in all_summaries.items():
        plt.plot(summary['length'], summary['n_itemsets'],
                 'o-', label=f'min_support={ms}')
    plt.yscale('log')
    plt.xlabel('Tamanho do itemset (nº de produtos)')
    plt.ylabel('Nº de itemsets (escala log)')
    plt.title('Distribuição do tamanho dos itemsets por min_support')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fpgrowth_length_distribution.png',
                dpi=120, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 05 — FP-Growth: itemsets frequentes")
    log.info("=" * 60)
    log.info(f"N_SAMPLE_ORDERS={N_SAMPLE_ORDERS:,}")
    log.info(f"min_support testados: {SUPPORTS}")
    log_ram("início")

    df = load_encoded()

    results = []
    all_summaries = {}

    for ms in SUPPORTS:
        itemsets, dt = run_fpgrowth(df, ms)

        if itemsets.empty:
            log.warning(f"  Nenhum itemset encontrado para min_support={ms}")
            results.append({
                'min_support': ms,
                'n_itemsets': 0,
                'time_s': dt,
                'n_items': 0,
                'max_length': 0,
            })
            continue

        itemsets, summary = describe_itemsets(itemsets)
        all_summaries[ms] = summary

        # log resumo por tamanho
        log.info("  Distribuição por tamanho:")
        for _, row in summary.iterrows():
            log.info(f"    len={int(row['length'])}: "
                     f"{int(row['n_itemsets']):>8,} itemsets | "
                     f"support {row['support_min']:.4f}–{row['support_max']:.4f}")

        # log top 10 itemsets por suporte (mais frequentes)
        top10 = itemsets.nlargest(10, 'support')[['support', 'itemsets']]
        log.info("  Top 10 itemsets por suporte:")
        for _, row in top10.iterrows():
            items_str = ', '.join(sorted(row['itemsets']))
            log.info(f"    {row['support']:.4f} | {items_str[:160]}"
                     f"{'...' if len(items_str) > 160 else ''}")

        # salva
        save_itemsets(itemsets, ms)

        results.append({
            'min_support': ms,
            'n_itemsets': len(itemsets),
            'time_s': dt,
            'n_items': df.shape[1],
            'max_length': int(itemsets['length'].max()),
        })

        del itemsets
        gc.collect()

    # ------- Resumo final -------
    log.info("=" * 60)
    log.info("Resumo comparativo:")
    res_df = pd.DataFrame(results)
    for _, row in res_df.iterrows():
        log.info(f"  min_support={row['min_support']:<6} | "
                 f"itemsets={int(row['n_itemsets']):>10,} | "
                 f"tempo={row['time_s']:.2f}s | "
                 f"tamanho_max={int(row['max_length'])}")

    res_df.to_csv(TAB_DIR / 'fpgrowth_summary.csv', index=False)
    log.info(f"Resumo salvo em {TAB_DIR / 'fpgrowth_summary.csv'}")

    # ------- Gráficos -------
    if len(results) > 1:
        plot_support_vs_itemsets(results)
        plot_length_distribution(all_summaries)
        log.info(f"Gráficos salvos em {FIG_DIR}")

    log_ram("fim")
    log.info("STEP 05 concluído com sucesso.")


if __name__ == '__main__':
    main()
