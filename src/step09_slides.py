# src/step09_slides.py
# Gera apresentação PPTX automática
# Rode com:  python -m src.step09_slides
# ============================================================

import sys
from pathlib import Path
from datetime import datetime

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from src.config import RESULTS, FIG_DIR, TAB_DIR, N_SAMPLE_ORDERS, INTEGRANTES
from src.utils import get_logger

log = get_logger('step09', 'step09_slides.log')

SLIDES_DIR = RESULTS / 'slides'
SLIDES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Paleta e constantes
# ------------------------------------------------------------
AZUL_ESCURO  = RGBColor(0x1A, 0x3D, 0x6B)
AZUL_MEDIO   = RGBColor(0x2E, 0x6D, 0xA4)
CINZA_TEXTO  = RGBColor(0x33, 0x33, 0x33)
CINZA_CLARO  = RGBColor(0xF0, 0xF4, 0xF8)
VERMELHO     = RGBColor(0xC0, 0x39, 0x2B)
BRANCO       = RGBColor(0xFF, 0xFF, 0xFF)

FONT_TITULO  = 'Calibri Light'
FONT_CORPO   = 'Calibri'

# Tamanhos (16:9 = 13.333 x 7.5 polegadas)
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ------------------------------------------------------------
# Helpers de construção
# ------------------------------------------------------------
def add_textbox(slide, left, top, width, height,
                text='', font_size=18, bold=False,
                color=CINZA_TEXTO, align=PP_ALIGN.LEFT,
                font_name=FONT_CORPO, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(font_size)
    r.font.bold = bold
    r.font.name = font_name
    r.font.color.rgb = color
    return tb


def add_bullets(slide, left, top, width, height,
                items, font_size=16, color=CINZA_TEXTO,
                bullet_color=AZUL_MEDIO):
    """items: list[str] ou list[(str, int_level)]"""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0

        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level

        # bullet marker
        bullet = '▪ ' if level == 0 else '– '
        r1 = p.add_run()
        r1.text = bullet
        r1.font.size = Pt(font_size)
        r1.font.name = FONT_CORPO
        r1.font.color.rgb = bullet_color
        r1.font.bold = True

        r2 = p.add_run()
        r2.text = text
        r2.font.size = Pt(font_size)
        r2.font.name = FONT_CORPO
        r2.font.color.rgb = color

    return tb


def add_title_bar(slide, title: str,
                  subtitle: str | None = None,
                  bg=AZUL_ESCURO, fg=BRANCO):
    """Barra de título retangular no topo."""
    bar_h = Inches(1.1)
    bar = slide.shapes.add_shape(
        1,  # rectangle
        Emu(0), Emu(0), SLIDE_W, bar_h
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = bg
    bar.line.fill.background()

    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_top = Inches(0.15)
    tf.margin_bottom = Inches(0.15)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.name = FONT_TITULO
    r.font.color.rgb = fg

    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        r2.font.size = Pt(14)
        r2.font.name = FONT_CORPO
        r2.font.color.rgb = fg


def add_footer(slide, idx: int, total: int):
    """Rodapé com numeração."""
    tb = slide.shapes.add_textbox(
        SLIDE_W - Inches(2), SLIDE_H - Inches(0.4),
        Inches(1.8), Inches(0.35)
    )
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"{idx} / {total}"
    r.font.size = Pt(10)
    r.font.name = FONT_CORPO
    r.font.color.rgb = RGBColor(0x99, 0x99, 0x99)


def add_table(slide, left, top, width, height,
              df: pd.DataFrame, font_size=11,
              header_color=AZUL_MEDIO):
    """Adiciona tabela a partir de DataFrame."""
    rows = len(df) + 1
    cols = len(df.columns)
    shape = slide.shapes.add_table(
        rows, cols, left, top, width, height
    )
    table = shape.table

    # header
    for c, col_name in enumerate(df.columns):
        cell = table.cell(0, c)
        cell.text = str(col_name)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(font_size)
            r.font.bold = True
            r.font.name = FONT_CORPO
            r.font.color.rgb = BRANCO

    # body
    for r_idx, (_, row) in enumerate(df.iterrows(), start=1):
        for c_idx, col in enumerate(df.columns):
            cell = table.cell(r_idx, c_idx)
            v = row[col]
            if isinstance(v, float):
                cell.text = f'{v:.4f}'
            else:
                cell.text = str(v)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (
                CINZA_CLARO if r_idx % 2 == 1 else BRANCO
            )
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            for run in p.runs:
                run.font.size = Pt(font_size)
                run.font.name = FONT_CORPO
                run.font.color.rgb = CINZA_TEXTO

    return shape


def add_image_fit(slide, path: Path, left, top, width=None, height=None):
    """Insere imagem mantendo aspect ratio dentro da caixa."""
    if not path.exists():
        log.warning(f"  imagem não encontrada: {path.name}")
        return None
    return slide.shapes.add_picture(
        str(path), left, top, width=width, height=height
    )


def new_slide(prs) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


# ------------------------------------------------------------
# Slides
# ------------------------------------------------------------
def slide_01_titulo(prs, total):
    s = new_slide(prs)

    # fundo azul
    bg = s.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = AZUL_ESCURO
    bg.line.fill.background()

    add_textbox(s, Inches(1), Inches(1.5), Inches(11), Inches(1.2),
                'Mineração de Dados',
                font_size=42, bold=True, color=BRANCO,
                font_name=FONT_TITULO)

    add_textbox(s, Inches(1), Inches(2.6), Inches(11), Inches(1),
                'Regras de Associação no Dataset Instacart',
                font_size=26, color=RGBColor(0xCC, 0xDD, 0xEE),
                font_name=FONT_TITULO)

    add_textbox(s, Inches(1), Inches(3.8), Inches(11), Inches(0.5),
                'Trabalho Prático — Avaliação do Bimestre',
                font_size=16, color=BRANCO)

    # --- INTEGRANTES ---
    add_textbox(s, Inches(1), Inches(4.5), Inches(11), Inches(0.4),
                'Integrantes:',
                font_size=14, bold=True, color=RGBColor(0xAA, 0xCC, 0xEE))

    add_bullets(s, Inches(1), Inches(4.9), Inches(11), Inches(1.2),
                INTEGRANTES,
                font_size=15, color=BRANCO,
                bullet_color=RGBColor(0x88, 0xBB, 0xFF))

    add_textbox(s, Inches(1), Inches(6.2), Inches(11), Inches(0.5),
                'Algoritmo: FP-Growth  |  Dataset: Instacart Market Basket Analysis',
                font_size=13, color=RGBColor(0xAA, 0xCC, 0xEE))

    add_textbox(s, Inches(1), Inches(6.7), Inches(11), Inches(0.4),
                f'Apresentação gerada em {datetime.now().strftime("%d/%m/%Y")}',
                font_size=10, color=RGBColor(0x88, 0xAA, 0xCC))

    add_footer(s, 1, total)


def slide_02_objetivo(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Objetivo e Problema de Negócio')

    add_textbox(s, Inches(0.6), Inches(1.4), Inches(12), Inches(0.5),
                'Objetivo', font_size=18, bold=True, color=AZUL_ESCURO)

    add_bullets(s, Inches(0.6), Inches(1.9), Inches(12), Inches(1.8),
                [
                    'Aplicar Mineração de Dados (KDD) com Regras de Associação em base real de grande porte.',
                    'Identificar padrões de compra conjunta em 3 níveis de granularidade.',
                    'Gerar insights acionáveis para o negócio.',
                ], font_size=15)

    add_textbox(s, Inches(0.6), Inches(3.9), Inches(12), Inches(0.5),
                'Problema de negócio', font_size=18, bold=True, color=AZUL_ESCURO)

    add_textbox(s, Inches(0.6), Inches(4.4), Inches(12), Inches(0.7),
                '“Quais produtos, corredores e departamentos são comprados juntos '
                'com frequência suficiente para gerar recomendações e estratégias de venda cruzada?”',
                font_size=14, color=AZUL_MEDIO)

    add_bullets(s, Inches(0.6), Inches(5.3), Inches(12), Inches(1.8),
                [
                    'Sistemas de recomendação ("quem levou X, também levou Y")',
                    'Layout físico de loja',
                    'Campanhas promocionais cruzadas',
                    'Gestão de sortimento',
                ], font_size=14)

    add_footer(s, 2, total)


def slide_03_base(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Base de Dados',
                  'Instacart Market Basket Analysis — Kaggle')

    dados = pd.DataFrame({
        'Métrica': [
            'Pedidos totais (orders)',
            'Itens em pedidos prior',
            'Produtos distintos',
            'Aisles (corredores)',
            'Departments',
            'Arquivos CSV',
        ],
        'Valor': [
            '3.421.083',
            '32.434.489',
            '49.688',
            '134',
            '21',
            '6',
        ],
    })

    add_table(s, Inches(0.6), Inches(1.5),
              Inches(12.1), Inches(3.5),
              dados, font_size=14)

    add_textbox(s, Inches(0.6), Inches(5.4), Inches(12.1), Inches(1.5),
                'Estrutura transacional: cada pedido = 1 cesta (lista de produtos comprados).',
                font_size=14, color=AZUL_ESCURO, bold=True)

    add_bullets(s, Inches(0.6), Inches(5.9), Inches(12.1), Inches(1.5),
                [
                    'Amostra: 400.000 pedidos (~12% do total)',
                    'Filtro: produtos com frequência ≥ 100 na amostra',
                ], font_size=13)

    add_footer(s, 3, total)


def slide_04_metodologia(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Metodologia — Pipeline')

    etapas = [
        ('1. EDA', 'Análise exploratória dos 6 arquivos CSV'),
        ('2. Amostragem', '400k pedidos aleatórios (seed=42)'),
        ('3. Transações', 'order_id → lista de produtos'),
        ('4. Filtro', 'produtos com freq ≥ 100 (5.992 produtos)'),
        ('5. TransactionEncoder', '3 granularidades: produto / aisle / department'),
        ('6. FP-Growth', 'min_support ajustado por granularidade'),
        ('7. Regras', 'association_rules com min_confidence e min_lift'),
        ('8. Insights', 'top regras + análise de negócio'),
    ]

    top = Inches(1.5)
    for i, (titulo, desc) in enumerate(etapas):
        y = top + Inches(i * 0.65)
        add_textbox(s, Inches(0.6), y, Inches(3.5), Inches(0.5),
                    titulo, font_size=14, bold=True, color=AZUL_MEDIO)
        add_textbox(s, Inches(4.2), y, Inches(8.5), Inches(0.5),
                    desc, font_size=13, color=CINZA_TEXTO)

    add_footer(s, 4, total)


def slide_05_eda(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Análise Exploratória — Top 20 Produtos')

    add_image_fit(s, FIG_DIR / 'top20_products.png',
                  Inches(0.8), Inches(1.4),
                  width=Inches(7.2))

    add_textbox(s, Inches(8.3), Inches(1.5), Inches(4.5), Inches(0.5),
                'Destaques', font_size=16, bold=True, color=AZUL_ESCURO)

    add_bullets(s, Inches(8.3), Inches(2.1), Inches(4.5), Inches(4.5),
                [
                    'Banana: 15,9% das cestas',
                    'Bag of Organic Bananas: 12,7%',
                    'Organic Strawberries: 9,0%',
                    'Organic Baby Spinach: 8,2%',
                    'Perfil dominante: frutas, vegetais e orgânicos',
                ], font_size=13)

    add_footer(s, 5, total)


def slide_06_prep(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Preparação dos Dados',
                  f'Amostra de {N_SAMPLE_ORDERS:,} pedidos')

    dados = pd.DataFrame({
        'Etapa': [
            'Pedidos amostrados',
            'Produtos válidos (freq ≥ 100)',
            'Cestas finais (≥ 2 produtos)',
            'Tamanho médio das cestas',
            'Total de itens preservados',
            'Cobertura de itens',
        ],
        'Resultado': [
            f'{N_SAMPLE_ORDERS:,}',
            '5.992',
            '368.250',
            '9,17 produtos',
            '3.378.090',
            '84,36%',
        ],
    })

    add_table(s, Inches(0.6), Inches(1.5),
              Inches(12.1), Inches(4.0),
              dados, font_size=14)

    add_textbox(s, Inches(0.6), Inches(6.0), Inches(12.1), Inches(1.0),
                'Justificativa: matriz booleana de 3,2M × 49k seria inviável em 16 GB RAM '
                '(~160 GB). A amostra preserva > 84% dos itens e captura os padrões fortes.',
                font_size=12, color=AZUL_MEDIO)

    add_footer(s, 6, total)


def slide_07_algoritmo(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Algoritmo: FP-Growth')

    add_textbox(s, Inches(0.6), Inches(1.4), Inches(12), Inches(0.5),
                'Por que FP-Growth?', font_size=18, bold=True, color=AZUL_ESCURO)

    dados = pd.DataFrame({
        'Aspecto': [
            'Estratégia',
            'Passagens pelo dataset',
            'Uso de memória',
            'Desempenho em bases grandes',
            'Nosso caso (368k × 5.991)',
        ],
        'Apriori': [
            'Gera candidatos',
            'k+1',
            'Alto',
            'Ruim',
            'inviável',
        ],
        'FP-Growth': [
            'FP-tree + recursão',
            '2',
            'Médio',
            'Excelente',
            '~55s, ~3 GB RAM',
        ],
    })

    add_table(s, Inches(0.6), Inches(2.0),
              Inches(12.1), Inches(3.5),
              dados, font_size=13)

    add_textbox(s, Inches(0.6), Inches(5.8), Inches(12.1), Inches(1.0),
                'FP-Growth faz apenas 2 passagens pela base e usa estrutura compacta '
                '(FP-tree), sendo ordens de magnitude mais rápido que o Apriori. '
                'GPU não traz ganho — algoritmo é essencialmente sequencial.',
                font_size=13, color=AZUL_MEDIO)

    add_footer(s, 7, total)


def slide_08_itemsets(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Itemsets Frequentes Gerados')

    dados = pd.DataFrame({
        'Granularidade': ['Produto', 'Aisle', 'Department'],
        'min_support':   ['0,001', '0,02', '0,05'],
        'Itemsets totais': ['4.413', '692', '242'],
        'Itemsets len ≥ 2': ['2.495', '627', '228'],
    })

    add_table(s, Inches(0.6), Inches(1.5),
              Inches(12.1), Inches(2.5),
              dados, font_size=14)

    add_image_fit(s, FIG_DIR / 'fpgrowth_produto_length_distribution.png',
                  Inches(3.5), Inches(4.2), width=Inches(6.3))

    add_footer(s, 8, total)


def _slide_regras(prs, total, idx, tag, titulo):
    s = new_slide(prs)
    add_title_bar(s, titulo)

    df = pd.read_csv(TAB_DIR / f'slide_top5_regras_{tag}.csv')
    show = df[['antecedents_str', 'consequents_str',
               'support', 'confidence', 'lift']].copy()
    show.columns = ['Antecedente', 'Consequente',
                    'Suporte', 'Confiança', 'Lift']

    # reduz tamanho do texto
    show['Antecedente'] = show['Antecedente'].apply(
        lambda s: s if len(s) <= 45 else s[:42] + '...'
    )
    show['Consequente'] = show['Consequente'].apply(
        lambda s: s if len(s) <= 45 else s[:42] + '...'
    )

    add_table(s, Inches(0.6), Inches(1.5),
              Inches(12.1), Inches(3.8),
              show, font_size=12)

    add_footer(s, idx, total)


def slide_09_regras_produto(prs, total):
    _slide_regras(prs, total, 9, 'produto',
                  'Top 5 Regras — Produto (Análise Micro)')


def slide_10_regras_aisle(prs, total):
    _slide_regras(prs, total, 10, 'aisle',
                  'Top 5 Regras — Aisle (Análise Média)')


def slide_11_regras_department(prs, total):
    _slide_regras(prs, total, 11, 'department',
                  'Top 5 Regras — Department (Análise Macro)')


def slide_12_insights(prs, total):
    s = new_slide(prs)
    add_title_bar(s, '5 Insights de Negócio')

    add_bullets(s, Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.5),
                [
                    ('Recomendação personalizada de produto: regras de alto lift '
                     '(ex.: alho + cebola, lift = 5,2) para sugerir complementos no carrinho.',
                     0),
                    ('Layout físico otimizado: aproximar corredores de frutas, '
                     'laticínios e vegetais embalados (lift ~2,7).', 0),
                    ('Promoções cruzadas por seção: combos deli + frozen + snacks '
                     'aproveitam confiança de 48%.', 0),
                    ('Foco em produtos frescos: perfil do cliente Instacart é fortemente '
                     'orientado a fresh produce.', 0),
                    ('Segmentação por cesta: clientes deli + frozen são "compradores completos" '
                     '— alvo ideal para upsell premium.', 0),
                ], font_size=15)

    add_footer(s, 12, total)


def slide_13_limitacoes(prs, total):
    s = new_slide(prs)
    add_title_bar(s, 'Limitações e Trabalhos Futuros')

    add_textbox(s, Inches(0.6), Inches(1.4), Inches(6), Inches(0.5),
                'Limitações', font_size=18, bold=True, color=VERMELHO)

    add_bullets(s, Inches(0.6), Inches(2.0), Inches(6), Inches(4.5),
                [
                    'Amostra de 400k pedidos (12%)',
                    'Filtro removeu produtos raros',
                    'Lift inflado em itens de baixo suporte',
                    'Análise transversal (sem tempo)',
                    'Sem dados demográficos',
                ], font_size=13)

    add_textbox(s, Inches(7), Inches(1.4), Inches(6), Inches(0.5),
                'Trabalhos Futuros', font_size=18, bold=True, color=AZUL_MEDIO)

    add_bullets(s, Inches(7), Inches(2.0), Inches(6), Inches(4.5),
                [
                    'Análise temporal (sazonalidade)',
                    'Segmentação por cluster de cliente',
                    'Regras sequenciais (PrefixSpan)',
                    'Comparação empírica Apriori vs FP-Growth',
                    'Deploy em motor de recomendação',
                ], font_size=13)

    add_footer(s, 13, total)

def slide_14_comparacao(prs, total):
    """Slide comparativo Apriori vs FP-Growth (evidência empírica)."""
    s = new_slide(prs)
    add_title_bar(s, 'Comparação Empírica: Apriori vs FP-Growth',
                  'mesma base, mesmo min_support = 0,005')

    # Gráfico principal (combined) à esquerda
    add_image_fit(s, FIG_DIR / 'step10_combined.png',
                  Inches(0.5), Inches(1.4),
                  width=Inches(8.0))

    # Números-chave à direita
    add_textbox(s, Inches(8.8), Inches(1.6), Inches(4.3), Inches(0.5),
                'Resultados', font_size=16, bold=True, color=AZUL_ESCURO)

    add_bullets(s, Inches(8.8), Inches(2.2), Inches(4.3), Inches(4.5),
                [
                    '5k cestas: 2,20x (FP mais rápido)',
                    '40k cestas: 1,16x',
                    '80k cestas: 1,62x',
                    ('160k cestas: 16,46x', 0),
                    ('FP-Growth: 14,7 s', 1),
                    ('Apriori: 242,0 s', 1),
                    ('Extrapolação para 368k:', 0),
                    ('Apriori seria inviável', 1),
                    ('FP-Growth: ~30-40 s', 1),
                ], font_size=12)

    add_footer(s, 14, total)

def slide_15_conclusao(prs, total):
    s = new_slide(prs)

    bg = s.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = AZUL_ESCURO
    bg.line.fill.background()

    add_textbox(s, Inches(1), Inches(1.5), Inches(11), Inches(1),
                'Conclusão', font_size=40, bold=True, color=BRANCO,
                font_name=FONT_TITULO)

    add_textbox(s, Inches(1), Inches(2.8), Inches(11), Inches(0.5),
                'FP-Growth aplicado com sucesso em 3 granularidades:',
                font_size=16, color=RGBColor(0xCC, 0xDD, 0xEE))

    add_bullets(s, Inches(1), Inches(3.5), Inches(11), Inches(3),
                [
                    'Produto: 4.413 itemsets → 144 regras confiáveis (lift até 5,2)',
                    'Aisle: 692 itemsets → 558 regras (lift até 2,7)',
                    'Department: 242 itemsets → 369 regras (conf. até 53%)',
                    'Padrão dominante: cesta saudável (frutas + laticínios + vegetais)',
                    'Insights acionáveis para recomendação, layout e promoções',
                ], font_size=14, color=BRANCO, bullet_color=RGBColor(0x88, 0xBB, 0xFF))

    add_footer(s, 15, total)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("STEP 09 — Geração de slides PPTX")
    log.info("=" * 60)

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    TOTAL = 15

    slide_01_titulo(prs, TOTAL)
    slide_02_objetivo(prs, TOTAL)
    slide_03_base(prs, TOTAL)
    slide_04_metodologia(prs, TOTAL)
    slide_05_eda(prs, TOTAL)
    slide_06_prep(prs, TOTAL)
    slide_07_algoritmo(prs, TOTAL)
    slide_08_itemsets(prs, TOTAL)
    slide_09_regras_produto(prs, TOTAL)
    slide_10_regras_aisle(prs, TOTAL)
    slide_11_regras_department(prs, TOTAL)
    slide_12_insights(prs, TOTAL)
    slide_13_limitacoes(prs, TOTAL)
    slide_14_comparacao(prs, TOTAL)
    slide_15_conclusao(prs, TOTAL)

    out = SLIDES_DIR / 'apresentacao.pptx'
    prs.save(str(out))
    size_kb = out.stat().st_size / 1024
    log.info(f"  Slides: {out}")
    log.info(f"  Tamanho: {size_kb:.1f} KB ({TOTAL} slides)")
    log.info("STEP 09 concluído.")


if __name__ == '__main__':
    main()
