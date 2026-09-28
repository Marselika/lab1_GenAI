"""
data_utils.py - Încărcarea corpusului, vocabularul la nivel de caracter,
împărțirea train/validation și construcția secvențelor (input, target).
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# 1. Încărcarea corpusului
# ---------------------------------------------------------------------------
def load_corpus(path: str | Path,
                normalization: str | None = None,
                max_chars: int | None = None) -> str:
    """
    Încarcă textul din:
      - un singur fișier .txt, sau
      - un folder: toate fișierele *.txt sunt concatenate (ordine alfabetică),
        separate printr-o linie goală -> un singur corpus.

    Textul NU este transformat (fără lowercase, fără eliminarea diacriticelor).
    Singurele efecte: BOM-ul UTF-8 este ignorat și finalurile de linie Windows
    (\\r\\n) devin \\n (comportamentul standard Python în mod text).
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"Niciun fișier .txt în folderul {p}")
        text = "\n\n".join(f.read_text(encoding="utf-8-sig") for f in files)
    elif p.is_file():
        text = p.read_text(encoding="utf-8-sig")  # strict: eroare clară dacă nu e UTF-8
    else:
        raise FileNotFoundError(
            f"Datasetul nu a fost găsit: {p}\n"
            f"Pune fișierul în folderul proiectului sau folosește --data <cale>."
        )

    if normalization:
        text = unicodedata.normalize(normalization, text)
    if max_chars:
        text = text[:max_chars]
    if not text:
        raise ValueError("Corpusul este gol.")
    return text


# ---------------------------------------------------------------------------
# 2. Vocabular la nivel de caracter
# ---------------------------------------------------------------------------
class Vocabulary:
    """Vocabular de caractere + mapările char_to_int / int_to_char."""

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.char_to_int: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char: dict[int, str] = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        # sorted(...) -> ordine deterministă, aceeași la fiecare rulare
        return cls(sorted(set(text)))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str, skip_unknown: bool = False) -> np.ndarray:
        """Text -> vector de întregi (int32). Caracterele necunoscute: eroare sau ignorate."""
        ids = []
        for ch in text:
            idx = self.char_to_int.get(ch)
            if idx is None:
                if skip_unknown:
                    continue
                raise KeyError(f"Caracter în afara vocabularului: {ch!r}")
            ids.append(idx)
        return np.asarray(ids, dtype=np.int32)

    def unknown_chars(self, text: str) -> list[str]:
        return sorted({ch for ch in text if ch not in self.char_to_int})

    def decode(self, ids) -> str:
        return "".join(self.int_to_char[int(i)] for i in ids)

    def save(self, path: str | Path, **meta) -> None:
        payload = {"chars": self.chars, **meta}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple["Vocabulary", dict]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        chars = payload.pop("chars")
        return cls(chars), payload


def printable_char(ch: str) -> str:
    """Reprezentare lizibilă pentru caractere invizibile (spațiu, \\n, \\t etc.)."""
    special = {" ": "␣ (spațiu)", "\n": "\\n", "\t": "\\t", "\r": "\\r", " ": "NBSP"}
    if ch in special:
        return special[ch]
    if unicodedata.category(ch)[0] in ("C", "Z"):
        return f"U+{ord(ch):04X}"
    return ch


def describe_corpus(text: str, vocab: Vocabulary, preview_chars: int = 500) -> dict:
    """Afișează statisticile cerute și returnează un dicționar cu ele."""
    counts = Counter(text)
    print("=" * 70)
    print("STATISTICI CORPUS")
    print("=" * 70)
    print(f"Număr total de caractere : {len(text):,}")
    print(f"Caractere unice (vocab)  : {vocab.size}")
    print(f"\nPrimele {preview_chars} caractere:\n" + "-" * 70)
    print(text[:preview_chars])
    print("-" * 70)
    print("Vocabular (index: caracter):")
    print("  " + "  ".join(f"{i}:{printable_char(c)}" for i, c in enumerate(vocab.chars)))
    print("\nCele mai frecvente 15 caractere:")
    for ch, n in counts.most_common(15):
        print(f"  {printable_char(ch):>12}  {n:>10,}  ({n / len(text):.2%})")
    rare = [c for c, n in counts.items() if n < 5]
    if rare:
        print(f"\nCaractere rare (<5 apariții): {len(rare)} -> "
              + " ".join(printable_char(c) for c in sorted(rare)[:40]))

    # Diagnostic util pentru română: ş/ţ (sedilă) vs ș/ț (virgulă) - doar semnalăm.
    if ("ş" in counts or "Ş" in counts) and ("ș" in counts or "Ș" in counts):
        print("\nATENȚIE: corpusul conține atât ş (sedilă) cât și ș (virgulă). "
              "Sunt caractere diferite pentru model. Nu le modific, doar semnalez.")
    if ("ţ" in counts or "Ţ" in counts) and ("ț" in counts or "Ț" in counts):
        print("ATENȚIE: corpusul conține atât ţ (sedilă) cât și ț (virgulă).")
    print("=" * 70)
    return {"total_chars": len(text), "unique_chars": vocab.size}


# ---------------------------------------------------------------------------
# 3. Împărțire train / validation FĂRĂ leakage
# ---------------------------------------------------------------------------
def train_val_split(encoded: np.ndarray, val_fraction: float, seq_length: int):
    """
    Împărțire CRONOLOGICĂ a textului codificat: primele (1 - val_fraction) din
    caractere -> train, restul -> validation.

    Ferestrele sunt construite SEPARAT în fiecare parte, deci nicio fereastră
    nu traversează granița. De ce nu amestecăm ferestrele înainte de split?
    Ferestrele consecutive (pas 1) au 99 din 100 de caractere în comun; un split
    aleator ar pune practic aceeași secvență și în train și în validation
    (leakage), iar validation loss ar fi artificial de mic.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction trebuie să fie în (0, 1)")
    split = int(len(encoded) * (1.0 - val_fraction))
    train, val = encoded[:split], encoded[split:]
    for name, part in (("train", train), ("validation", val)):
        if len(part) <= seq_length:
            raise ValueError(
                f"Partea de {name} are {len(part)} caractere, prea puțin pentru "
                f"seq_length={seq_length}. Micșorează seq_length sau mărește corpusul."
            )
    return train, val


# ---------------------------------------------------------------------------
# 4. Secvențe (input, target) ca tf.data.Dataset
# ---------------------------------------------------------------------------
def num_windows(n_chars: int, seq_length: int, step: int) -> int:
    # Fereastra care începe la i folosește pozițiile i .. i+seq_length (inclusiv),
    # deci i <= n_chars - seq_length - 1.
    return len(range(0, n_chars - seq_length, step))


def make_dataset(encoded: np.ndarray, seq_length: int, batch_size: int,
                 step: int = 1, shuffle: bool = False, seed: int | None = None,
                 target_mode: str = "many_to_one"):
    """
    Pentru fiecare poziție de start i, x = encoded[i : i + seq_length], iar:

      many_to_one : y = encoded[i + seq_length]             (următorul caracter)
                    batch: x (B, L) int32, y (B,) int32
      many_to_many: y = encoded[i + 1 : i + seq_length + 1] (x deplasat cu 1)
                    batch: x (B, L) int32, y (B, L) int32
                    -> la fiecare poziție t modelul prezice caracterul t+1,
                       deci o fereastră aduce seq_length targeturi, nu unul.
                       Ultimul target este exact targetul din many_to_one.

    Ferestrele NU sunt materializate în memorie (N x 101 întregi ar putea avea
    GB). Păstrăm doar indicii de start și tăiem ferestrele "din zbor" cu tf.gather.
    """
    if target_mode not in ("many_to_one", "many_to_many"):
        raise ValueError(f"target_mode necunoscut: {target_mode}")
    starts = np.arange(0, len(encoded) - seq_length, step, dtype=np.int64)
    if len(starts) == 0:
        raise ValueError("Nu se poate construi nicio secvență.")

    text_tensor = tf.constant(encoded, dtype=tf.int32)
    offsets = tf.range(seq_length + 1, dtype=tf.int64)  # 0..seq_length

    ds = tf.data.Dataset.from_tensor_slices(starts)
    if shuffle:
        ds = ds.shuffle(len(starts), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def to_xy(batch_starts):
        idx = batch_starts[:, None] + offsets[None, :]      # (B, seq_length + 1)
        windows = tf.gather(text_tensor, idx)               # (B, seq_length + 1)
        if target_mode == "many_to_many":
            return windows[:, :-1], windows[:, 1:]          # (B, L), (B, L)
        return windows[:, :-1], windows[:, -1]              # (B, L), (B,)

    ds = ds.map(to_xy, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds, len(starts)


def full_context_loss(model, val_ids: np.ndarray, seq_length: int,
                      step: int = 5, batch_size: int = 256) -> float:
    """
    Cross-entropy (nats) doar pe predicțiile făcute cu context COMPLET de
    seq_length caractere, pe ferestre de validare cu pasul `step`.

    Motiv: în many_to_many, val_loss din Keras face media peste toate pozițiile,
    inclusiv cele cu context scurt (poziția 0 vede un singur caracter), deci e
    ușor pesimist. Metrica de aici măsoară la fel ambele moduri și le face comparabile.
    """
    ds, _ = make_dataset(val_ids, seq_length, batch_size, step=step, shuffle=False,
                         target_mode="many_to_one")
    total, n = 0.0, 0
    for x, y in ds:
        probs = model(x, training=False)
        if len(probs.shape) == 3:          # many_to_many: (B, L, V) -> ultima poziție
            probs = probs[:, -1, :]
        p = tf.gather(probs, tf.cast(y, tf.int32), axis=1, batch_dims=1)
        total += float(-tf.reduce_sum(tf.math.log(tf.clip_by_value(p, 1e-12, 1.0))))
        n += int(y.shape[0])
    return total / n


def unigram_perplexity(train_ids: np.ndarray, val_ids: np.ndarray, vocab_size: int) -> float:
    """
    Baseline: un model care ignoră contextul și prezice mereu frecvența
    caracterelor din train (cu add-one smoothing). Modelul LSTM trebuie să
    obțină o perplexitate clar mai mică decât aceasta.
    """
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    return float(np.exp(-np.mean(np.log(probs[val_ids]))))
