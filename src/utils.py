# src/utils.py
# Funções auxiliares / utilitárias
# ============================================================

import time
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import LOG_DIR


# ------------------------------------------------------------
# Timer
# ------------------------------------------------------------
@contextmanager
def timer(label: str):
    t0 = time.time()
    print(f"[START] {label}")
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[DONE ] {label}  ({dt:.2f}s)")


# ------------------------------------------------------------
# Logger para arquivo + console
# ------------------------------------------------------------
def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is None:
        log_file = f"{name}.log"
    fh = logging.FileHandler(LOG_DIR / log_file, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ------------------------------------------------------------
# Formatação de tamanho
# ------------------------------------------------------------
def human_size(n_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"

# ------------------------------------------------------------
# Formatação de nomes
# ------------------------------------------------------------
def fmt_integrantes(nomes: list[str]) -> str:
    """
    Formata lista de nomes como 'A e B' ou 'A, B e C'.

    Exemplos:
        ['Carlos']                        -> 'Carlos'
        ['Carlos', 'Gustavo']             -> 'Carlos e Gustavo'
        ['Carlos', 'Gustavo', 'Maria']    -> 'Carlos, Gustavo e Maria'
    """
    nomes = [n.strip() for n in nomes if n and n.strip()]
    if not nomes:
        return ''
    if len(nomes) == 1:
        return nomes[0]
    if len(nomes) == 2:
        return f'{nomes[0]} e {nomes[1]}'
    return ', '.join(nomes[:-1]) + f' e {nomes[-1]}'
