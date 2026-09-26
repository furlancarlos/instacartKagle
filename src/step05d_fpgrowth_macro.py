# src/step05d_fpgrowth_macro.py
# FP-Growth em AISLE e DEPARTMENT
# Rode com:  python -m src.step05d_fpgrowth_macro
# ============================================================

import sys
import os
import gc
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import fpgrowth

from src.config import CACHE_DIR, FIG_DIR, TAB_DIR, N_SAMPLE_ORDERS
from src.utils import timer, get_logger

log = get_logger('step05d', 'step05d_fpgrowth_macro.log')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100


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
# Configuração dos pipelines
# ------------------------------------------------------------
PIPELINES = [
    {
        'name': 'aisle',
        'file': CACHE_DIR / f'encoded_aisle_{N_SAMPLE_ORDERS // 1000}k.parquet',
        'supports': [0.10, 0.05, 0.02],
    },
    {
        'name': 'department',
        'file': CACHE_DIR / f'encoded_department_{N_SAMPLE_ORDERS // 1000}k.parquet',
        'supports': [0.20, 0.10, 0.05],
    },
]


# ------------------------------------------------------------
# Funções
# ------------------------------------------------------------
def load_encoded(path: Path) -> pd.DataFrame:
    log.info(f"Carregando {path.name} ...")
    with timer(f"load {path.name}"):
        df = pd.read_parquet(path)
    log.info(f"  shape: {df.shape}  "
             f"| memória: {df.memory_usage(deep=True).sum() / (1024**2):,.1f} MB")
    return df


def run_fpgrowth(df: pd.DataFrame, min_support: float):
    log.info("-" * 60)
    log.info(f"FP-Growth min_support={min_support}")
    log_ram("antes")

    t0 = time.time()
    itemsets = fpgrowth(
        df,
        min_support=min_support,
        use_colnames=True,
        max_len=None,
        verbose=0,
    )
    dt = time.time() - t0

    log.info(f"  tempo: {dt:.2f}s | itemsets: {len(itemsets):,}")
    log_ram("depois")
    return itemsets, dt


def describe(itemsets: pd.DataFrame):
    itemsets = itemsets.copy()
    itemsets['length'] = itemsets['itemsets'].apply(len)
    summary = (
        itemsets.groupby('length')
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


def save_itemsets(itemsets: pd.DataFrame, min_support: float, tag: str) -> Path:
    fname = f"itemsets_{tag}_support{str(min_support).replace('.', '_')}.parquet"
    out = CACHE_DIR / fname
    out_df = itemsets.copy()
    out_df['itemsets'] = out_df['itemsets'].apply(lambda s: '||'.join(sorted(s)))
    out_df.to_parquet(out, index=False)
    log.info(f"  salvo: {out.name}  ({out.stat().st_size / (1024**2):,.2f} MB)")
    return out


def plot_support_vs_itemsets(results: list[dict], tag: str):
    df = pd.DataFrame(results)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('min_support')
    ax1.set_ylabel('Nº de itemsets', color='tab:blue')
    ax1.plot(df['min_support'], df['n_itemsets'], 'o-', color='tab:blue', lw=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Tempo (s)', color='tab:red')
    ax2.plot(df['min_support'], df['time_s'], 's--', color='tab:red', lw=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'FP-Growth [{tag}]: min_support × nº itemsets / tempo')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fpgrowth_{tag}_support_vs_itemsets.png',
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_length_distribution(all_summaries: dict, tag: str):
    plt.figure(figsize=(12, 6))
    for ms, summary in all_summaries.items():
        plt.plot(summary['length'], summary['n_itemsets'],
                 'o-', label=f'min_support={ms}')
    plt.yscale('log')
    plt.xlabel('Tamanho do itemset')
    plt.ylabel('Nº de itemsets (escala log)')
    plt.title(f'Distribuição do tamanho dos itemsets [{tag}]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fpgrowth_{tag}_length_distribution.png',
                dpi=120, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 05d — FP-Growth em AISLE e DEPARTMENT")
    log.info("=" * 60)
    log_ram("início")

    all_results = {}

    for cfg in PIPELINES:
        tag = cfg['name']
        log.info("=" * 60)
        log.info(f"Pipeline: {tag.upper()}")
        log.info(f"  supports: {cfg['supports']}")
        log.info("=" * 60)

        df = load_encoded(cfg['file'])
        log_ram(f"após carregar {tag}")

        results = []
        all_summaries = {}

        for ms in cfg['supports']:
            itemsets, dt = run_fpgrowth(df, ms)

            if itemsets.empty:
                log.warning(f"  nenhum itemset para min_support={ms}")
                results.append({'min_support': ms, 'n_itemsets': 0,
                                'time_s': dt, 'max_length': 0,
                                'n_len_ge_2': 0})
                continue

            itemsets, summary = describe(itemsets)
            all_summaries[ms] = summary

            log.info("  Distribuição por tamanho:")
            for _, row in summary.iterrows():
                log.info(f"    len={int(row['length'])}: "
                         f"{int(row['n_itemsets']):>8,} itemsets | "
                         f"support {row['support_min']:.4f}–{row['support_max']:.4f}")

            multi = itemsets[itemsets['length'] >= 2].nlargest(10, 'support')
            log.info("  Top 10 itemsets (len>=2):")
            for _, row in multi.iterrows():
                items_str = ', '.join(sorted(row['itemsets']))
                log.info(f"    {row['support']:.4f} | {items_str[:150]}")

            save_itemsets(itemsets, ms, tag)

            results.append({
                'min_support': ms,
                'n_itemsets': len(itemsets),
                'time_s': dt,
                'max_length': int(itemsets['length'].max()),
                'n_len_ge_2': int((itemsets['length'] >= 2).sum()),
            })

            del itemsets
            gc.collect()

        log.info("=" * 60)
        log.info(f"Resumo [{tag}]:")
        res_df = pd.DataFrame(results)
        for _, row in res_df.iterrows():
            log.info(f"  min_support={row['min_support']:<6} | "
                     f"itemsets={int(row['n_itemsets']):>8,} | "
                     f"len>=2={int(row.get('n_len_ge_2', 0)):>8,} | "
                     f"tempo={row['time_s']:.2f}s | "
                     f"max_len={int(row['max_length'])}")

        res_df.to_csv(TAB_DIR / f'fpgrowth_{tag}_summary.csv', index=False)

        if len(results) > 1:
            plot_support_vs_itemsets(results, tag)
            plot_length_distribution(all_summaries, tag)

        all_results[tag] = results
        del df
        gc.collect()

    log_ram("fim")
    log.info("STEP 05d concluído.")


if __name__ == '__main__':
    main()
