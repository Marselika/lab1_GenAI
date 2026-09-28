"""
generate.py - Generare de text caracter-cu-caracter cu sampling pe temperatură.

Utilizare din linia de comandă:
    python generate.py --seed "România este" --num-chars 400 --temperature 0.5
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import ARTIFACTS_DIR, DEFAULT_NUM_CHARS

MODEL_FILE = "model.keras"
VOCAB_FILE = "vocab.json"


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Re-scalează distribuția modelului:
        q_i = exp(log p_i / T) / sum_j exp(log p_j / T)   (echivalent: p_i^(1/T) normalizat)

    T < 1 -> distribuție mai "ascuțită" (conservator); T > 1 -> mai "plată" (creativ);
    T = 1 -> distribuția originală a modelului.
    """
    if temperature <= 0:
        raise ValueError("Temperatura trebuie să fie > 0.")
    p = np.asarray(probs, dtype=np.float64)
    logits = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    logits -= logits.max()              # stabilitate numerică (evită overflow la T mic)
    q = np.exp(logits)
    return q / q.sum()                  # suma exact 1 în float64 -> OK pt. rng.choice


def sample_with_temperature(probs: np.ndarray, temperature: float,
                            rng: np.random.Generator) -> tuple[int, np.ndarray]:
    q = apply_temperature(probs, temperature)
    return int(rng.choice(len(q), p=q)), q


def entropy_bits(q: np.ndarray) -> float:
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


@dataclass
class GenerationResult:
    seed_used: str            # seed-ul efectiv (fără caractere necunoscute)
    generated: str            # doar caracterele noi
    unknown_chars: list[str]  # caractere din seed care nu sunt în vocabular
    mean_entropy_bits: float  # entropia medie a distribuției după temperatură

    @property
    def full_text(self) -> str:
        return self.seed_used + self.generated


class CharGenerator:
    """Împachetează modelul + vocabularul și implementează bucla de generare."""

    def __init__(self, model, vocab, seq_length: int):
        import tensorflow as tf

        self.model = model
        self.vocab = vocab
        self.seq_length = int(seq_length)
        # tf.function cu formă (None, None): o singură compilare a grafului pentru
        # orice lungime de context -> mult mai rapid decât model.predict() în buclă.
        self._forward = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32)],
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path = ARTIFACTS_DIR) -> "CharGenerator":
        import keras
        from data_utils import Vocabulary

        model_dir = Path(model_dir)
        model_path, vocab_path = model_dir / MODEL_FILE, model_dir / VOCAB_FILE
        if not model_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                f"Nu găsesc {model_path} și/sau {vocab_path}. Rulează întâi: python train.py"
            )
        model = keras.models.load_model(model_path)
        vocab, meta = Vocabulary.load(vocab_path)
        # Verificare de consistență model <-> vocabular
        out_dim = model.output_shape[-1]
        if out_dim != vocab.size:
            raise ValueError(f"Model cu {out_dim} ieșiri, dar vocabular cu {vocab.size} caractere.")
        return cls(model, vocab, meta["seq_length"])

    def next_char_probs(self, context_ids: list[int]) -> np.ndarray:
        x = np.asarray(context_ids[-self.seq_length:], dtype=np.int32)[None, :]  # (1, T)
        out = self._forward(x).numpy()[0]      # many_to_one: (V,) | many_to_many: (T, V)
        return out[-1] if out.ndim == 2 else out  # distribuția pentru caracterul următor

    def generate(self, seed_text: str, num_chars: int = DEFAULT_NUM_CHARS,
                 temperature: float = 1.0, rng_seed: int | None = None) -> GenerationResult:
        if num_chars < 0:
            raise ValueError("num_chars trebuie să fie >= 0")
        # Dacă un caracter lipsește din vocabular dar varianta lui mică există
        # (ex. corpus doar cu litere mici: "R" -> "r"), folosim varianta mică.
        seed_text = "".join(
            c if c in self.vocab.char_to_int or c.lower() not in self.vocab.char_to_int
            else c.lower() for c in seed_text)
        unknown = self.vocab.unknown_chars(seed_text)
        ids = self.vocab.encode(seed_text, skip_unknown=True).tolist()
        if not ids:
            raise ValueError("Seed-ul nu conține niciun caracter din vocabularul modelului.")
        seed_used = self.vocab.decode(ids)

        rng = np.random.default_rng(rng_seed)
        out_chars, entropies = [], []
        for _ in range(num_chars):
            # Fereastră glisantă: exact ca la antrenare, modelul vede ultimele
            # seq_length caractere și pornește dintr-o stare LSTM nulă.
            probs = self.next_char_probs(ids)
            next_id, q = sample_with_temperature(probs, temperature, rng)
            entropies.append(entropy_bits(q))
            ids.append(next_id)
            out_chars.append(self.vocab.int_to_char[next_id])

        return GenerationResult(
            seed_used=seed_used,
            generated="".join(out_chars),
            unknown_chars=unknown,
            mean_entropy_bits=float(np.mean(entropies)) if entropies else math.nan,
        )


# ---------------------------------------------------------------------------
# Funcția cerută: generate_text(seed_text, num_chars, temperature)
# ---------------------------------------------------------------------------
_default_generator: CharGenerator | None = None


def generate_text(seed_text: str, num_chars: int = DEFAULT_NUM_CHARS,
                  temperature: float = 1.0, rng_seed: int | None = None,
                  model_dir: str | Path = ARTIFACTS_DIR) -> str:
    """Returnează seed-ul + `num_chars` caractere generate. Modelul se încarcă o singură dată."""
    global _default_generator
    if _default_generator is None:
        _default_generator = CharGenerator.from_dir(model_dir)
    return _default_generator.generate(seed_text, num_chars, temperature, rng_seed).full_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generează text cu modelul LSTM antrenat.")
    parser.add_argument("--seed", required=True, help="Textul inițial")
    parser.add_argument("--num-chars", type=int, default=DEFAULT_NUM_CHARS)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--rng-seed", type=int, default=None, help="Pentru rezultate reproductibile")
    parser.add_argument("--model-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()

    gen = CharGenerator.from_dir(args.model_dir)
    res = gen.generate(args.seed, args.num_chars, args.temperature, args.rng_seed)
    if res.unknown_chars:
        print(f"[Atenție] Caractere ignorate (lipsă din vocabular): {res.unknown_chars}")
    print(res.full_text)


if __name__ == "__main__":
    main()
