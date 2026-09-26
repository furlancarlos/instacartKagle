# src/step10_apriori_vs_fpgrowth.py
# Comparação empírica: Apriori vs FP-Growth
# Rode com:  python -m src.step10_apriori_vs_fpgrowth
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
from mlxtend.frequent_patterns import apriori, fpgrowth

from src.config import CACHE_DIR, FIG_DIR, TAB_DIR, RANDOM_SEED
from src.utils import timer, get_logger

log = get_logger('step10', 'step10_comparacao.log')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200


try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def ram_mb() -> float:
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# ------------------------------------------------------------
# Configuração das rodadas
# ------------------------------------------------------------
# Rodada 1: min_support baixo (0.002) em tamanhos pequenos/médios
ROUND_1 = {
    'name': 'round1_support_0_005',
    'min_support': 0.005,
    'sizes': [5_000, 10_000, 20_000, 40_000],
    'algos': ['fpgrowth', 'apriori'],
}

# Rodada 2: min_support alto (0.01) em tamanhos grandes
ROUND_2 = {
    'name': 'round2_support_0_005',
    'min_support': 0.005,
    'sizes': [80_000, 160_000],
    'algos': ['fpgrowth', 'apriori'],
}

ROUNDS = [ROUND_1, ROUND_2]

MAX_TIME_PER_RUN = 300   # teto de 5 min por execução


# ------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------
def load_encoded() -> pd.DataFrame:
    path = CACHE_DIR / 'encoded_400k.parquet'
    log.info(f"Carregando {path.name} ...")
    with timer("load encoded"):
        df = pd.read_parquet(path)
    log.info(f"  shape: {df.shape}")
    return df


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    idx.sort()
    return df.iloc[idx].reset_index(drop=True)


def run_algorithm(df: pd.DataFrame, algo: str, min_support: float):
    """Roda fpgrowth ou apriori. Retorna (n_itemsets, tempo, ram_delta)."""
    ram_before = ram_mb()
    t0 = time.time()

    if algo == 'fpgrowth':
        itemsets = fpgrowth(
            df, min_support=min_support,
            use_colnames=True, max_len=None, verbose=0,
        )
    elif algo == 'apriori':
        itemsets = apriori(
            df, min_support=min_support,
            use_colnames=True, max_len=None, verbose=0,
        )
    else:
        raise ValueError(f"algoritmo desconhecido: {algo}")

    dt = time.time() - t0
    ram_after = ram_mb()
    ram_delta = max(ram_after - ram_before, 0.0)
    n_itemsets = len(itemsets)

    del itemsets
    gc.collect()
    return n_itemsets, dt, ram_delta


# ------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------
def plot_time_by_round(res_df: pd.DataFrame,
                       round_cfg: dict,
                       suffix: str = '') -> None:
    """Gráfico tempo x tamanho, para uma rodada."""
    df = res_df[res_df['round'] == round_cfg['name']]
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Tempo
    ax = axes[0]
    for algo, color in [('fpgrowth', '#2E86AB'), ('apriori', '#C0392B')]:
        sub = df[df['algo'] == algo].sort_values('n_orders')
        ax.plot(sub['n_orders'], sub['time_s'], 'o-',
                label=algo.capitalize(), color=color, lw=2, markersize=8)
    ax.set_xlabel('Nº de cestas')
    ax.set_ylabel('Tempo (s)')
    ax.set_title(f"Tempo — min_support = {round_cfg['min_support']}",
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Itemsets
    ax = axes[1]
    for algo, color in [('fpgrowth', '#2E86AB'), ('apriori', '#C0392B')]:
        sub = df[df['algo'] == algo].sort_values('n_orders')
        ax.plot(sub['n_orders'], sub['n_itemsets'], 's-',
                label=algo.capitalize(), color=color, lw=2, markersize=8)
    ax.set_xlabel('Nº de cestas')
    ax.set_ylabel('Nº de itemsets')
    ax.set_title('Itemsets gerados (iguais nos dois)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.suptitle(f"Rodada: {round_cfg['name']}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'step10_time_{suffix or round_cfg["name"]}.png',
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_combined(res_df: pd.DataFrame) -> None:
    """Compara as 2 rodadas em um único gráfico de tempo."""
    plt.figure(figsize=(13, 6))

    for round_cfg, marker, style in [
        (ROUND_1, 'o', '-'),
        (ROUND_2, '^', '--'),
    ]:
        df = res_df[res_df['round'] == round_cfg['name']]
        for algo, color in [('fpgrowth', '#2E86AB'), ('apriori', '#C0392B')]:
            sub = df[df['algo'] == algo].sort_values('n_orders')
            if sub.empty:
                continue
            label = f"{algo.capitalize()} (sup={round_cfg['min_support']})"
            plt.plot(sub['n_orders'], sub['time_s'],
                     marker=marker, linestyle=style,
                     color=color, lw=2, markersize=8, label=label)

    plt.xlabel('Nº de cestas')
    plt.ylabel('Tempo (s)')
    plt.title('Comparação Apriori vs FP-Growth (2 rodadas)',
              fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step10_combined.png',
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_speedup(res_df: pd.DataFrame) -> None:
    """Speedup (Apriori / FP-Growth) por tamanho e rodada."""
    pivot = res_df.pivot_table(
        index=['round', 'n_orders'],
        columns='algo',
        values='time_s'
    ).reset_index()

    pivot['speedup'] = pivot['apriori'] / pivot['fpgrowth']

    fig, ax = plt.subplots(figsize=(12, 6))

    for round_cfg, color in [
        (ROUND_1, '#8E44AD'),
        (ROUND_2, '#16A085'),
    ]:
        sub = pivot[pivot['round'] == round_cfg['name']]
        ax.plot(sub['n_orders'], sub['speedup'], 'o-',
                color=color, lw=2.5, markersize=10,
                label=f"sup={round_cfg['min_support']}")

    ax.axhline(y=1.0, color='black', linestyle=':', lw=1.5,
               label='Empate (speedup=1)')
    ax.set_xlabel('Nº de cestas')
    ax.set_ylabel('Speedup (tempo Apriori / tempo FP-Growth)')
    ax.set_title('Onde o FP-Growth passa o Apriori?\n'
                 '(acima de 1 = Apriori mais rápido; abaixo = FP-Growth mais rápido)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'step10_speedup.png',
                dpi=200, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 10 — Comparação empírica Apriori vs FP-Growth")
    log.info("=" * 60)
    log.info(f"Rodadas: {[r['name'] for r in ROUNDS]}")
    log.info(f"Teto de tempo por execução: {MAX_TIME_PER_RUN}s")
    log.info("")

    df_full = load_encoded()

    results = []

    for round_cfg in ROUNDS:
        log.info("=" * 70)
        log.info(f"RODADA: {round_cfg['name']}")
        log.info(f"  min_support = {round_cfg['min_support']}")
        log.info(f"  tamanhos    = {round_cfg['sizes']}")
        log.info("=" * 70)

        for n in round_cfg['sizes']:
            log.info("-" * 60)
            log.info(f"  Amostra: {n:,} cestas")
            log.info("-" * 60)

            df = subsample(df_full, n, RANDOM_SEED)
            log.info(f"    shape: {df.shape}")

            for algo in round_cfg['algos']:
                log.info(f"    --- {algo.upper()} ---")
                try:
                    n_itemsets, dt, ram_delta = run_algorithm(
                        df, algo, round_cfg['min_support']
                    )
                    log.info(f"      itemsets: {n_itemsets:,}")
                    log.info(f"      tempo   : {dt:.2f}s")
                    log.info(f"      RAM delta: {ram_delta:,.1f} MB")

                    results.append({
                        'round': round_cfg['name'],
                        'min_support': round_cfg['min_support'],
                        'n_orders': n,
                        'algo': algo,
                        'n_itemsets': n_itemsets,
                        'time_s': dt,
                        'ram_delta_mb': ram_delta,
                    })

                    if dt > MAX_TIME_PER_RUN:
                        log.warning(
                            f"      ⚠ excedeu {MAX_TIME_PER_RUN}s — "
                            f"as próximas execuções desta rodada continuam, "
                            f"mas considere reduzir amostras."
                        )

                except MemoryError:
                    log.error(f"      MemoryError com {n:,} cestas!")
                    results.append({
                        'round': round_cfg['name'],
                        'min_support': round_cfg['min_support'],
                        'n_orders': n,
                        'algo': algo,
                        'n_itemsets': -1,
                        'time_s': -1,
                        'ram_delta_mb': -1,
                    })
                except Exception as e:
                    log.error(f"      erro em {algo}: {e!r}")
                    results.append({
                        'round': round_cfg['name'],
                        'min_support': round_cfg['min_support'],
                        'n_orders': n,
                        'algo': algo,
                        'n_itemsets': -1,
                        'time_s': -1,
                        'ram_delta_mb': -1,
                    })

            del df
            gc.collect()

    # --------------------------------------------------------
    # Consolidado
    # --------------------------------------------------------
    res_df = pd.DataFrame(results)
    res_df.to_csv(TAB_DIR / 'step10_apriori_vs_fpgrowth.csv',
                  index=False, float_format='%.4f')
    log.info("=" * 60)
    log.info("Resultado consolidado:")
    log.info("=" * 60)

    for round_cfg in ROUNDS:
        sub = res_df[res_df['round'] == round_cfg['name']]
        if sub.empty:
            continue
        log.info(f"\nRodada: {round_cfg['name']} "
                 f"(min_support={round_cfg['min_support']})")
        for n in round_cfg['sizes']:
            s = sub[sub['n_orders'] == n]
            fp = s[s['algo'] == 'fpgrowth']
            ap = s[s['algo'] == 'apriori']
            if fp.empty or ap.empty:
                continue
            t_fp = fp.iloc[0]['time_s']
            t_ap = ap.iloc[0]['time_s']
            it_fp = fp.iloc[0]['n_itemsets']
            it_ap = ap.iloc[0]['n_itemsets']
            speedup = t_ap / t_fp if t_fp > 0 else 0
            vencedor = 'Apriori' if speedup > 1 else 'FP-Growth'
            log.info(f"  {n:>7,} cestas | "
                     f"FP: {t_fp:>7.2f}s ({it_fp:>6,} it) | "
                     f"AP: {t_ap:>7.2f}s ({it_ap:>6,} it) | "
                     f"speedup: {speedup:>5.2f}x ({vencedor})")

    # --------------------------------------------------------
    # Gráficos
    # --------------------------------------------------------
    if len(res_df) >= 4:
        plot_time_by_round(res_df, ROUND_1, suffix='round1')
        plot_time_by_round(res_df, ROUND_2, suffix='round2')
        plot_combined(res_df)
        plot_speedup(res_df)
        log.info(f"\nGráficos salvos em {FIG_DIR}")

    log.info("STEP 10 concluído.")


if __name__ == '__main__':
    main()
