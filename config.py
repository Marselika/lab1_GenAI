"""
config.py - Hiperparametrii și căile proiectului, într-un singur loc.

Modifică valorile de mai jos, sau suprascrie-le din linia de comandă
(`python train.py --help` arată opțiunile).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Căi
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "corpus_ro_clean.txt"   # fișier .txt SAU folder cu .txt-uri
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"          # model, vocabular, grafice, metrici

# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------
SEQ_LENGTH = 100          # lungimea secvenței de input (caractere)
TARGET_MODE = "many_to_many"
# "many_to_many": target = secvența deplasată cu 1 poziție -> 100 de predicții
#                 antrenate per fereastră (recomandat, de ~50-100x mai rapid pe epocă).
# "many_to_one" : target = doar caracterul de după fereastră (varianta clasică).
STEP = None               # pasul dintre începuturile ferestrelor; None = automat:
                          # SEQ_LENGTH pentru many_to_many, 1 pentru many_to_one
VAL_FRACTION = 0.1        # ultimele 10% din corpus -> validare
MAX_CHARS = None          # ex. 2_000_000 pentru experimente rapide; None = tot corpusul
UNICODE_NORMALIZATION = None  # None = textul rămâne exact cum e; opțional "NFC"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 128
LSTM_UNITS = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

# ---------------------------------------------------------------------------
# Antrenare
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
CLIPNORM = 1.0            # gradient clipping (protecție contra exploding gradients)
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-5
SAMPLE_EVERY_N_EPOCHS = 5  # afișează un text generat la fiecare N epoci (0 = dezactivat)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Generare
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURES = (0.2, 0.5, 1.0, 1.5)
DEFAULT_NUM_CHARS = 400


@dataclass
class TrainConfig:
    """Toți parametrii unei rulări de antrenare (serializabili în JSON)."""
    data_path: str = str(DATA_PATH)
    output_dir: str = str(ARTIFACTS_DIR)
    seq_length: int = SEQ_LENGTH
    target_mode: str = TARGET_MODE
    step: int | None = STEP
    val_fraction: float = VAL_FRACTION
    max_chars: int | None = MAX_CHARS
    unicode_normalization: str | None = UNICODE_NORMALIZATION
    embedding_dim: int = EMBEDDING_DIM
    lstm_units: int = LSTM_UNITS
    num_lstm_layers: int = NUM_LSTM_LAYERS
    dropout: float = DROPOUT
    batch_size: int = BATCH_SIZE
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    clipnorm: float = CLIPNORM
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    reduce_lr_patience: int = REDUCE_LR_PATIENCE
    reduce_lr_factor: float = REDUCE_LR_FACTOR
    min_lr: float = MIN_LR
    sample_every_n_epochs: int = SAMPLE_EVERY_N_EPOCHS
    seed: int = RANDOM_SEED

    def resolved_step(self) -> int:
        if self.step is not None:
            return self.step
        return self.seq_length if self.target_mode == "many_to_many" else 1

    def to_dict(self) -> dict:
        return asdict(self)
