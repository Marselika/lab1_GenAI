"""
evaluate.py - Evaluarea modelului: grafice loss/accuracy/perplexitate,
perplexitate + bits-per-character și diagnostic overfitting/underfitting.

Rulare independentă (după train.py), regenerează graficele și raportul:
    python evaluate.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fără fereastră (merge și pe server / în Streamlit)
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


def perplexity(ce_loss_nats: float) -> float:
    """PP = exp(H), unde H = cross-entropy medie în nats (loss-ul Keras)."""
    return float(math.exp(ce_loss_nats))


def bits_per_char(ce_loss_nats: float) -> float:
    return float(ce_loss_nats / math.log(2))


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)


def plot_history(history: dict, out_dir: str | Path) -> list[Path]:
    """Salvează loss.png, accuracy.png, perplexity.png în out_dir."""
    out_dir = Path(out_dir)
    epochs = range(1, len(history["loss"]) + 1)
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["loss"], color=TRAIN_COLOR, lw=2, marker="o", ms=4, label="train loss")
    ax.plot(epochs, history["val_loss"], color=VAL_COLOR, lw=2, ls="--", marker="s", ms=4,
            label="validation loss")
    _style(ax, "Evoluția loss-ului (cross-entropy)", "Loss (nats / caracter)")
    fig.tight_layout(); p = out_dir / "loss.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    if "accuracy" in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, history["accuracy"], color=TRAIN_COLOR, lw=2, marker="o", ms=4,
                label="train accuracy")
        ax.plot(epochs, history["val_accuracy"], color=VAL_COLOR, lw=2, ls="--", marker="s", ms=4,
                label="validation accuracy")
        _style(ax, "Acuratețea top-1 a predicției următorului caracter", "Accuracy")
        fig.tight_layout(); p = out_dir / "accuracy.png"; fig.savefig(p, dpi=150); plt.close(fig)
        paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [perplexity(v) for v in history["val_loss"]], color=VAL_COLOR, lw=2,
            ls="--", marker="s", ms=4, label="validation perplexity")
    _style(ax, "Perplexitatea pe validare", "Perplexity")
    fig.tight_layout(); p = out_dir / "perplexity.png"; fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)
    return paths


def diagnose(history: dict, unigram_ppl: float | None = None) -> list[str]:
    """
    Diagnostic EURISTIC (pragurile sunt orientative, nu reguli absolute).

    Notă: train loss este calculat cu Dropout ACTIV, deci e ușor supraestimat;
    un val loss puțin mai mic decât train loss este normal la început.
    """
    tl, vl = history["loss"], history["val_loss"]
    n = len(vl)
    best = min(range(n), key=lambda i: vl[i])
    msgs = [f"Cea mai bună epocă: {best + 1}/{n} (val_loss = {vl[best]:.4f}, "
            f"perplexity = {perplexity(vl[best]):.2f})."]

    gap = vl[-1] - tl[-1]
    rising = (n - 1 - best) >= 2 and vl[-1] > vl[best] * 1.02
    if rising or gap > 0.15 * tl[-1]:
        msgs.append(
            f"OVERFITTING probabil: val_loss a crescut după epoca {best + 1} și/sau diferența "
            f"val-train la final este {gap:+.3f}. Remedii: mai mult dropout, model mai mic, "
            f"mai multe date; EarlyStopping a restaurat deja cele mai bune ponderi."
        )
    else:
        msgs.append(f"Fără semne clare de overfitting (diferența val-train la final: {gap:+.3f}).")

    still_improving = n >= 2 and (tl[-2] - tl[-1]) / tl[-2] > 0.01 and best == n - 1
    weak = unigram_ppl is not None and perplexity(vl[best]) > 0.6 * unigram_ppl
    if weak:
        msgs.append(
            f"UNDERFITTING: perplexitatea ({perplexity(vl[best]):.2f}) este apropiată de baseline-ul "
            f"unigram ({unigram_ppl:.2f}); modelul folosește puțin contextul. Remedii: mai multe "
            f"epoci, mai multe unități LSTM, learning rate verificat."
        )
    elif still_improving:
        msgs.append(
            "Modelul încă se îmbunătățea la ultima epocă (train și val loss în scădere): "
            "antrenarea nu a convers - mai multe epoci ar ajuta (ușor underfitting)."
        )
    else:
        msgs.append("Fără semne clare de underfitting.")
    return msgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-generează graficele și raportul de evaluare.")
    parser.add_argument("--model-dir", default=str(ARTIFACTS_DIR))
    args = parser.parse_args()
    d = Path(args.model_dir)
    history = json.loads((d / "history.json").read_text(encoding="utf-8"))
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    for p in plot_history(history, d):
        print(f"Grafic salvat: {p}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "diagnosis"}, indent=2, ensure_ascii=False))
    for m in diagnose(history, metrics.get("unigram_perplexity")):
        print("-", m)


if __name__ == "__main__":
    main()
