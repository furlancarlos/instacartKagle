# src/step07_insights.py
# Interpretação + Insights + Material slides
# Rode com:  python -m src.step07_insights
# ============================================================

import sys
import os
import gc
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CACHE_DIR, FIG_DIR, TAB_DIR
from src.utils import timer, get_logger

log = get_logger('step07', 'step07_insights.log')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150   # alta resolução para slides
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 11


# ------------------------------------------------------------
# Configuração dos pipelines
# ------------------------------------------------------------
PIPELINES = [
    {
        'name': 'produto',
        'file': CACHE_DIR / 'rules_produto.parquet',
        'filters': {'support': 0.005, 'lift': 1.5, 'confidence': 0.05},
    },
    {
        'name': 'aisle',
        'file': CACHE_DIR / 'rules_aisle.parquet',
        'filters': {'support': 0.0, 'lift': 1.5, 'confidence': 0.30},
    },
    {
        'name': 'department',
        'file': CACHE_DIR / 'rules_department.parquet',
        'filters': {'support': 0.0, 'lift': 1.2, 'confidence': 0.40},
    },
]


# ------------------------------------------------------------
# Carregamento
# ------------------------------------------------------------
def load_rules(path: Path) -> pd.DataFrame:
    log.info(f"Carregando {path.name} ...")
    df = pd.read_parquet(path)
    # reconstrói frozensets
    df['antecedents'] = df['antecedents'].apply(
        lambda s: frozenset(s.split('||')) if isinstance(s, str) else s
    )
    df['consequents'] = df['consequents'].apply(
        lambda s: frozenset(s.split('||')) if isinstance(s, str) else s
    )
    log.info(f"  regras: {len(df):,}")
    return df


def apply_filters(rules: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = (
        (rules['support'] >= filters['support']) &
        (rules['lift'] >= filters['lift']) &
        (rules['confidence'] >= filters['confidence'])
    )
    out = rules[mask].copy()
    log.info(f"  após filtros {filters}: {len(out):,} regras")
    return out


# ------------------------------------------------------------
# Formatação
# ------------------------------------------------------------
def fmt_set(s) -> str:
    """'Banana, Organic Blueberries' — ordenado."""
    return ', '.join(sorted(s))


def make_top_table(rules: pd.DataFrame, n: int = 5,
                   sort_by: str = 'lift') -> pd.DataFrame:
    top = rules.nlargest(n, sort_by).copy()
    top['antecedents_str'] = top['antecedents'].apply(fmt_set)
    top['consequents_str'] = top['consequents'].apply(fmt_set)
    top['n_ant'] = top['antecedents'].apply(len)
    top['n_con'] = top['consequents'].apply(len)
    cols = [
        'antecedents_str', 'consequents_str',
        'support', 'confidence', 'lift',
        'n_ant', 'n_con',
    ]
    return top[cols].reset_index(drop=True)


# ------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------
def plot_top_rules_bar(top_df: pd.DataFrame, tag: str) -> None:
    """Barra horizontal: top 5/10 regras por lift."""
    df = top_df.copy()
    df['rule'] = df['antecedents_str'] + '  →  ' + df['consequents_str']
    # trunca para não estourar
    df['rule_short'] = df['rule'].apply(
        lambda s: s if len(s) <= 90 else s[:87] + '...'
    )
    df = df.sort_values('lift', ascending=True)

    plt.figure(figsize=(13, 0.55 * len(df) + 2.5))
    bars = plt.barh(df['rule_short'], df['lift'],
                    color=plt.cm.viridis(
                        np.linspace(0.2, 0.9, len(df))
                    ))
    plt.xlabel('Lift')
    plt.title(f'Top {len(df)} regras [{tag}] por Lift',
              fontsize=13, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, df['lift']):
        plt.text(bar.get_width() + 0.02 * df['lift'].max(),
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.2f}',
                 va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'slide_top{len(df)}_regras_{tag}.png',
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_scatter(rules: pd.DataFrame, tag: str,
                 max_points: int = 5000) -> None:
    """Scatter support × confidence, cor=lift."""
    df = rules
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    plt.figure(figsize=(11, 7))
    sc = plt.scatter(
        df['support'], df['confidence'],
        c=df['lift'], cmap='plasma',
        s=22, alpha=0.65, edgecolors='none'
    )
    plt.colorbar(sc, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title(f'Regras [{tag}] — support × confidence (cor=lift)\n'
              f'n={len(rules):,}', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'slide_scatter_{tag}.png',
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap_aisle(rules_aisle: pd.DataFrame) -> None:
    """
    Heatmap de co-ocorrência: eixo X = consequents (aisles únicos),
    eixo Y = antecedents (aisles únicos). Usa lift.
    Só funciona bem com aisle/department (poucos itens únicos).
    """
    # mantém apenas regras 1-a-1 (1 antecedente, 1 consequente)
    r = rules_aisle[
        (rules_aisle['antecedents'].apply(len) == 1) &
        (rules_aisle['consequents'].apply(len) == 1)
    ].copy()
    if r.empty:
        return

    r['ant'] = r['antecedents'].apply(lambda s: next(iter(s)))
    r['con'] = r['consequents'].apply(lambda s: next(iter(s)))

    # pega os aisles mais frequentes (por aparição no par)
    top_items = pd.Series(
        list(r['ant']) + list(r['con'])
    ).value_counts().head(20).index.tolist()

    r = r[r['ant'].isin(top_items) & r['con'].isin(top_items)]

    pivot = r.pivot_table(
        index='ant', columns='con', values='lift', aggfunc='mean'
    )
    pivot = pivot.reindex(index=top_items, columns=top_items)

    plt.figure(figsize=(13, 11))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False,
                cbar_kws={'label': 'Lift'}, linewidths=0.4)
    plt.title('Heatmap de co-ocorrência entre Aisles (lift)\n'
              'Top 20 aisles mais frequentes', fontsize=13,
              fontweight='bold')
    plt.xlabel('Consequente')
    plt.ylabel('Antecedente')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'slide_heatmap_aisle.png',
                dpi=200, bbox_inches='tight')
    plt.close()


def plot_heatmap_department(rules_dept: pd.DataFrame) -> None:
    r = rules_dept[
        (rules_dept['antecedents'].apply(len) == 1) &
        (rules_dept['consequents'].apply(len) == 1)
    ].copy()
    if r.empty:
        return

    r['ant'] = r['antecedents'].apply(lambda s: next(iter(s)))
    r['con'] = r['consequents'].apply(lambda s: next(iter(s)))

    pivot = r.pivot_table(
        index='ant', columns='con', values='lift', aggfunc='mean'
    )
    # ordena por frequência
    order = pd.Series(
        list(r['ant']) + list(r['con'])
    ).value_counts().index.tolist()
    pivot = pivot.reindex(index=order, columns=order)

    plt.figure(figsize=(11, 9))
    sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt='.2f',
                cbar_kws={'label': 'Lift'}, linewidths=0.4,
                annot_kws={'fontsize': 9})
    plt.title('Heatmap de co-ocorrência entre Departamentos (lift)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Consequente')
    plt.ylabel('Antecedente')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'slide_heatmap_department.png',
                dpi=200, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------
# Insights automáticos (heurística + dados)
# ------------------------------------------------------------
def generate_insights(tag: str, top_df: pd.DataFrame) -> list[str]:
    """Gera insights baseados nas top regras — texto pronto para slides."""
    insights = []

    if tag == 'produto':
        for _, row in top_df.head(5).iterrows():
            ant = row['antecedents_str']
            con = row['consequents_str']
            sup = row['support'] * 100
            conf = row['confidence'] * 100
            lift = row['lift']
            insights.append(
                f"Quem compra [{ant}] tem {conf:.1f}% de chance de comprar "
                f"[{con}] (lift={lift:.2f}). "
                f"Aparece em {sup:.2f}% das cestas."
            )
        insights.append(
            "INSIGHT DE NEGÓCIO: usar essas regras para recomendar produtos "
            "complementares no carrinho (\"quem levou X, também levou Y\"), "
            "aumentando o ticket médio em categorias específicas."
        )

    elif tag == 'aisle':
        for _, row in top_df.head(5).iterrows():
            ant = row['antecedents_str']
            con = row['consequents_str']
            conf = row['confidence'] * 100
            lift = row['lift']
            insights.append(
                f"Clientes que compram em [{ant}] também compram em "
                f"[{con}] em {conf:.1f}% dos casos (lift={lift:.2f})."
            )
        insights.append(
            "INSIGHT DE NEGÓCIO: posicionar fisicamente esses corredores "
            "próximos (ex.: frutas frescas perto de laticínios e vegetais "
            "embalados) aumenta a chance de compra cruzada."
        )

    elif tag == 'department':
        for _, row in top_df.head(5).iterrows():
            ant = row['antecedents_str']
            con = row['consequents_str']
            conf = row['confidence'] * 100
            lift = row['lift']
            insights.append(
                f"Seções [{ant}] e [{con}] são fortemente associadas "
                f"({conf:.1f}% de confiança, lift={lift:.2f})."
            )
        insights.append(
            "INSIGHT DE NEGÓCIO: campanhas promocionais cruzadas entre "
            "seções (ex.: 'leve massa e ganhe desconto em enlatados') "
            "aproveitam o padrão de compra observado."
        )

    return insights


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 07 — Interpretação + Insights + Material slides")
    log.info("=" * 60)

    all_tops = {}

    for cfg in PIPELINES:
        tag = cfg['name']
        log.info("=" * 60)
        log.info(f"Pipeline: {tag.upper()}")
        log.info("=" * 60)

        rules = load_rules(cfg['file'])
        rules_filtered = apply_filters(rules, cfg['filters'])

        if rules_filtered.empty:
            log.warning(f"  nenhuma regra para {tag} após filtros!")
            continue

        # top 5 por lift
        top5 = make_top_table(rules_filtered, n=5, sort_by='lift')
        all_tops[tag] = top5

        # log detalhado
        log.info("  TOP 5 regras (por lift):")
        for i, row in top5.iterrows():
            log.info(f"    [{i+1}] {row['antecedents_str']}  →  "
                     f"{row['consequents_str']}")
            log.info(f"         support={row['support']:.4f}  "
                     f"confidence={row['confidence']:.4f}  "
                     f"lift={row['lift']:.3f}")

        # insights
        insights = generate_insights(tag, top5)
        log.info(f"  Insights gerados ({len(insights)}):")
        for i, ins in enumerate(insights, 1):
            log.info(f"    [{i}] {ins}")

        # salva CSV top 5
        csv_out = TAB_DIR / f'slide_top5_regras_{tag}.csv'
        top5.to_csv(csv_out, index=False, float_format='%.4f')
        log.info(f"  CSV: {csv_out.name}")

        # top 10 para gráfico
        top10 = make_top_table(rules_filtered, n=10, sort_by='lift')
        csv10 = TAB_DIR / f'slide_top10_regras_{tag}.csv'
        top10.to_csv(csv10, index=False, float_format='%.4f')

        # gráficos
        plot_top_rules_bar(top10, tag)
        plot_scatter(rules_filtered, tag)

        # heatmaps específicos
        if tag == 'aisle':
            plot_heatmap_aisle(rules_filtered)
        elif tag == 'department':
            plot_heatmap_department(rules_filtered)

        gc.collect()

    log.info("=" * 60)
    log.info("MATERIAL GERADO — resumo")
    log.info("=" * 60)
    log.info(f"  CSVs (tables/): slide_top5_regras_*.csv, slide_top10_regras_*.csv")
    log.info(f"  Gráficos (figures/): slide_*.png")
    log.info("STEP 07 concluído.")


if __name__ == '__main__':
    main()
