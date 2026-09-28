"""
experiments.py - Experimente comparative.

1) Temperaturi (folosește modelul deja antrenat):
    python experiments.py temperature --seed "Istoria României" --num-chars 300

2) Lungimi de secvență (antrenează câte un model mic pentru fiecare valoare):
    python experiments.py seq-length --values 50 100 150 --epochs 5 --max-chars 1000000
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import keras

from config import ARTIFACTS_DIR, DEFAULT_TEMPERATURES, TrainConfig

WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------
# Metrici simple pentru textul generat
# ---------------------------------------------------------------------------
def distinct_ngram_ratio(text: str, n: int = 4) -> float:
    """Proporția de n-grame de caractere distincte. Mic = text repetitiv."""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def known_word_ratio(text: str, known_words: set[str] | None) -> float | None:
    """Proporția de cuvinte generate care există în corpus. Mic = cuvinte inventate."""
    if not known_words:
        return None
    words = WORD_RE.findall(text.lower())
    return sum(w in known_words for w in words) / len(words) if words else 0.0


def load_known_words(data_path: str | Path) -> set[str] | None:
    try:
        from data_utils import load_corpus
        return set(WORD_RE.findall(load_corpus(data_path).lower()))
    except FileNotFoundError:
        return None


def compare_temperatures(generator, seed_text: str, num_chars: int,
                         temperatures=DEFAULT_TEMPERATURES, rng_seed: int = 123,
                         known_words: set[str] | None = None) -> list[dict]:
    """Același seed + același rng_seed pentru fiecare T -> singura diferență e temperatura."""
    rows = []
    for t in temperatures:
        res = generator.generate(seed_text, num_chars, t, rng_seed=rng_seed)
        rows.append({
            "temperature": t,
            "text": res.full_text,
            "generated": res.generated,
            "mean_entropy_bits": res.mean_entropy_bits,
            "distinct_4gram_ratio": distinct_ngram_ratio(res.generated, 4),
            "known_word_ratio": known_word_ratio(res.generated, known_words),
        })
    return rows


def run_temperature_experiment(args) -> None:
    from generate import CharGenerator

    gen = CharGenerator.from_dir(args.model_dir)
    cfg = json.loads((Path(args.model_dir) / "train_config.json").read_text(encoding="utf-8"))
    known = load_known_words(cfg["data_path"])
    rows = compare_temperatures(gen, args.seed, args.num_chars, args.temperatures,
                                args.rng_seed, known)

    lines = [f"# Experiment temperatură\n\nSeed: `{args.seed}`  |  caractere generate: {args.num_chars}\n"]
    for r in rows:
        kw = "n/a" if r["known_word_ratio"] is None else f"{r['known_word_ratio']:.1%}"
        header = (f"T = {r['temperature']}  |  entropie medie = {r['mean_entropy_bits']:.2f} biți  |  "
                  f"4-grame distincte = {r['distinct_4gram_ratio']:.1%}  |  cuvinte existente = {kw}")
        print("\n" + "=" * 90 + f"\n{header}\n" + "=" * 90 + f"\n{r['text']}")
        lines.append(f"## {header}\n\n```text\n{r['text']}\n```\n")
    out = Path(args.model_dir) / "temperature_experiment.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRezultate salvate în {out}")


def run_seq_length_experiment(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from train import train

    base = TrainConfig(data_path=args.data, epochs=args.epochs, max_chars=args.max_chars,
                       sample_every_n_epochs=0)
    root = Path(args.model_dir) / "experiments"
    results = []
    for L in args.values:
        print(f"\n{'#' * 70}\n# seq_length = {L}\n{'#' * 70}")
        keras.backend.clear_session()
        cfg = replace(base, seq_length=L, output_dir=str(root / f"seq_{L}"))
        m = train(cfg)
        hist = json.loads((Path(cfg.output_dir) / "history.json").read_text(encoding="utf-8"))
        results.append({"seq_length": L, "val_loss": m["val_loss"], "val_perplexity": m["val_perplexity"],
                         "val_accuracy": m["val_accuracy"], "epochs_run": m["epochs_run"],
                         "train_seconds": m["train_seconds"], "val_loss_curve": hist["val_loss"]})

    print("\n" + "=" * 70 + "\nCOMPARAȚIE seq_length\n" + "=" * 70)
    print(f"{'L':>6} {'val_loss':>10} {'perplexity':>11} {'val_acc':>9} {'epoci':>6} {'timp(s)':>9}")
    for r in results:
        print(f"{r['seq_length']:>6} {r['val_loss']:>10.4f} {r['val_perplexity']:>11.3f} "
              f"{r['val_accuracy']:>9.4f} {r['epochs_run']:>6} {r['train_seconds']:>9.1f}")
    (root / "seq_length_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(results):
        ax.plot(range(1, len(r["val_loss_curve"]) + 1), r["val_loss_curve"], lw=2, marker="o", ms=4,
                color=colors[i % len(colors)], ls=styles[i % len(styles)], label=f"seq_length={r['seq_length']}")
    ax.set_title("Validation loss pentru diferite lungimi de secvență", loc="left")
    ax.set_xlabel("Epoca"); ax.set_ylabel("Validation loss")
    ax.grid(True, color="#e5e5e5"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(root / "seq_length_comparison.png", dpi=150)
    print(f"Grafic: {root / 'seq_length_comparison.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Experimente: temperatură și lungime de secvență.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("temperature")
    t.add_argument("--seed", required=True)
    t.add_argument("--num-chars", type=int, default=300)
    t.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES))
    t.add_argument("--rng-seed", type=int, default=123)
    t.add_argument("--model-dir", default=str(ARTIFACTS_DIR))

    s = sub.add_parser("seq-length")
    s.add_argument("--values", type=int, nargs="+", default=[50, 100, 150])
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--max-chars", type=int, default=1_000_000)
    s.add_argument("--data", default=TrainConfig().data_path)
    s.add_argument("--model-dir", default=str(ARTIFACTS_DIR))

    args = p.parse_args()
    run_temperature_experiment(args) if args.cmd == "temperature" else run_seq_length_experiment(args)


if __name__ == "__main__":
    main()
