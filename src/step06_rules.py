# src/step06_rules.py
# Regras de Associação (3 granularidades)
# Rode com:  python -m src.step06_rules
# ============================================================

import sys
import os
import gc
import time
from pathlib import Path
from itertools import product

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import association_rules

from src.config import CACHE_DIR, FIG_DIR, TAB_DIR
from src.utils import timer, get_logger

log = get_logger('step06', 'step06_rules.log')
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
        'name': 'produto',
        'itemsets_file': CACHE_DIR / 'itemsets_produto_support0_001.parquet',
        'min_confidence': [0.02, 0.05, 0.10],
        'min_lift':       [1.0, 1.5, 2.0],
    },
    {
        'name': 'aisle',
        'itemsets_file': CACHE_DIR / 'itemsets_aisle_support0_02.parquet',
        'min_confidence': [0.20, 0.30, 0.40],
        'min_lift':       [1.0, 1.2, 1.5],
    },
    {
        'name': 'department',
        'itemsets_file': CACHE_DIR / 'itemsets_department_support0_05.parquet',
        'min_confidence': [0.30, 0.40, 0.50],
        'min_lift':       [1.0, 1.1, 1.2],
    },
]


# ------------------------------------------------------------
# Carregar itemsets
# ------------------------------------------------------------
def load_itemsets(path: Path) -> pd.DataFrame:
    log.info(f"Carregando itemsets de {path.name} ...")
    with timer(f"load {path.name}"):
        df = pd.read_parquet(path)
        # reconstrói frozenset a partir de '||'.join
        df['itemsets'] = df['itemsets'].apply(
            lambda s: frozenset(s.split('||'))
        )
    log.info(f"  itemsets: {len(df):,}")
    return df


# ------------------------------------------------------------
# Gerar regras
# ------------------------------------------------------------
def generate_rules(itemsets: pd.DataFrame,
                   min_confidence: float,
                   min_lift: float) -> pd.DataFrame:
    t0 = time.time()
    rules = association_rules(
        itemsets,
        metric='confidence',
        min_threshold=min_confidence,
        num_itemsets=len(itemsets),   # obrigatório em mlxtend >= 0.24
    )
    # filtro adicional por lift
    rules = rules[rules['lift'] >= min_lift].copy()
    dt = time.time() - t0
    return rules, dt


def format_rules_for_save(rules: pd.DataFrame) -> pd.DataFrame:
    """Converte frozensets em strings legíveis para salvar."""
    out = rules.copy()
    out['antecedents'] = out['antecedents'].apply(lambda s: '||'.join(sorted(s)))
    out['consequents'] = out['consequents'].apply(lambda s: '||'.join(sorted(s)))
    return out


def top_rules(rules: pd.DataFrame, n: int = 5,
              sort_by: str = 'lift') -> pd.DataFrame:
    return rules.nlargest(n, sort_by)[
        ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    ]


# ------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------
def plot_rules_scatter(rules: pd.DataFrame, tag: str,
                       min_conf: float, min_lift: float,
                       max_points: int = 5000) -> None:
    """Scatter support × confidence, cor = lift."""
    df = rules
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    plt.figure(figsize=(11, 7))
    sc = plt.scatter(
        df['support'], df['confidence'],
        c=df['lift'], cmap='viridis',
        s=18, alpha=0.65, edgecolors='none'
    )
    plt.colorbar(sc, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title(f'Regras [{tag}] — support × confidence\n'
              f'(min_conf={min_conf}, min_lift={min_lift}, n={len(rules):,})')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'rules_{tag}_support_conf_lift.png',
                dpi=120, bbox_inches='tight')
    plt.close()


def plot_param_heatmap(results: list[dict], tag: str) -> None:
    """Heatmap: min_confidence × min_lift → nº de regras."""
    df = pd.DataFrame(results)
    pivot = df.pivot_table(
        index='min_lift', columns='min_confidence',
        values='n_rules', aggfunc='mean'
    ).sort_index(ascending=False)

    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu',
                cbar_kws={'label': 'Nº de regras'})
    plt.title(f'Regras [{tag}] — nº de regras por parâmetro')
    plt.xlabel('min_confidence')
    plt.ylabel('min_lift')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'rules_{tag}_param_heatmap.png',
                dpi=120, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Pipeline por granularidade
# ------------------------------------------------------------
def process_pipeline(cfg: dict) -> dict:
    tag = cfg['name']
    log.info("=" * 60)
    log.info(f"Pipeline: {tag.upper()}")
    log.info(f"  itemsets base: {cfg['itemsets_file'].name}")
    log.info(f"  min_confidence testados: {cfg['min_confidence']}")
    log.info(f"  min_lift testados:       {cfg['min_lift']}")
    log.info("=" * 60)

    itemsets = load_itemsets(cfg['itemsets_file'])

    # Considera só itemsets de tamanho >= 2 (não geram regras sozinhos)
    itemsets_multi = itemsets[itemsets['itemsets'].apply(len) >= 2].copy()
    log.info(f"  itemsets com len>=2: {len(itemsets_multi):,}")

    results = []
    best_rules_by_lift = None
    best_cfg = None

    for mc, ml in product(cfg['min_confidence'], cfg['min_lift']):
        # Use o DataFrame completo, não apenas os de tamanho >= 2
        rules, dt = generate_rules(itemsets, mc, ml)

        n = len(rules)
        log.info(f"  min_conf={mc:<6} | min_lift={ml:<5} | "
                 f"regras={n:>8,} | tempo={dt:.2f}s")

        if n > 0:
            log.info(f"    confidence: min={rules['confidence'].min():.3f} "
                     f"mediana={rules['confidence'].median():.3f} "
                     f"max={rules['confidence'].max():.3f}")
            log.info(f"    lift      : min={rules['lift'].min():.3f} "
                     f"mediana={rules['lift'].median():.3f} "
                     f"max={rules['lift'].max():.3f}")

        results.append({
            'min_confidence': mc,
            'min_lift': ml,
            'n_rules': n,
            'time_s': dt,
            'conf_median': float(rules['confidence'].median()) if n else np.nan,
            'lift_median': float(rules['lift'].median()) if n else np.nan,
            'lift_max':    float(rules['lift'].max()) if n else np.nan,
        })

        # Guarda o conjunto "melhor" = maior lift máximo (com pelo menos 20 regras)
        if n >= 20 and (best_rules_by_lift is None
                        or rules['lift'].max() > best_rules_by_lift['lift'].max()):
            best_rules_by_lift = rules.copy()
            best_cfg = (mc, ml)

    # Salva tabela de resultados
    res_df = pd.DataFrame(results)
    res_df.to_csv(TAB_DIR / f'rules_{tag}_param_search.csv', index=False)
    log.info(f"  tabela de busca: rules_{tag}_param_search.csv")

    # Visualizações
    if len(results) > 1:
        plot_param_heatmap(results, tag)

    # Seleciona o "melhor" conjunto para análise
    if best_rules_by_lift is not None and len(best_rules_by_lift) > 0:
        mc, ml = best_cfg
        log.info(f"  >>> Melhor conjunto: min_conf={mc}, min_lift={ml} "
                 f"({len(best_rules_by_lift):,} regras)")

        # Scatter
        plot_rules_scatter(best_rules_by_lift, tag, mc, ml)

        # Top 5 por lift
        top5 = top_rules(best_rules_by_lift, n=5, sort_by='lift')
        log.info("  Top 5 regras por LIFT:")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            ant = ', '.join(sorted(row['antecedents']))
            con = ', '.join(sorted(row['consequents']))
            log.info(f"    [{i}] {ant} -> {con} | "
                     f"sup={row['support']:.4f} conf={row['confidence']:.4f} "
                     f"lift={row['lift']:.3f}")

        # Salva regras completas em parquet
        best_out = CACHE_DIR / f'rules_{tag}.parquet'
        format_rules_for_save(best_rules_by_lift).to_parquet(
            best_out, index=False
        )
        log.info(f"  regras salvas: {best_out.name}")

        # Salva top regras em CSV
        top_all = top_rules(best_rules_by_lift, n=50, sort_by='lift')
        top_all_out = TAB_DIR / f'top50_regras_{tag}.csv'
        top_all.to_csv(top_all_out, index=False)
        log.info(f"  top 50 (CSV): {top_all_out.name}")

        return {
            'tag': tag,
            'best_cfg': best_cfg,
            'n_rules_best': len(best_rules_by_lift),
            'rules': best_rules_by_lift,
            'results': results,
        }

    log.warning(f"  Nenhum conjunto com >=20 regras em [{tag}]")
    return {'tag': tag, 'best_cfg': None, 'n_rules_best': 0,
            'rules': None, 'results': results}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 06 — Regras de Associação")
    log.info("=" * 60)
    log_ram("início")

    summaries = []

    for cfg in PIPELINES:
        out = process_pipeline(cfg)
        summaries.append(out)
        gc.collect()

    # Resumo final
    log.info("=" * 60)
    log.info("Resumo geral:")
    log.info("=" * 60)
    for s in summaries:
        if s['best_cfg'] is None:
            log.info(f"  [{s['tag']:>10}] nenhuma regra significativa")
        else:
            mc, ml = s['best_cfg']
            log.info(f"  [{s['tag']:>10}] best: min_conf={mc} min_lift={ml} | "
                     f"regras={s['n_rules_best']:,}")

    log_ram("fim")
    log.info("STEP 06 concluído.")


if __name__ == '__main__':
    main()
