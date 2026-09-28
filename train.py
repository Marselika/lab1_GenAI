"""
train.py - Pipeline complet: încărcare corpus -> vocabular -> secvențe ->
model LSTM -> antrenare -> salvare -> evaluare.

    python train.py                     # parametrii din config.py
    python train.py --epochs 5 --max-chars 500000   # rulare rapidă de test
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import keras
import numpy as np

from config import TrainConfig
from data_utils import (Vocabulary, describe_corpus, full_context_loss, load_corpus,
                        make_dataset, train_val_split, unigram_perplexity)
from evaluate import bits_per_char, diagnose, perplexity, plot_history
from generate import MODEL_FILE, VOCAB_FILE, CharGenerator
from model import build_model


class SampleTextCallback(keras.callbacks.Callback):
    """Afișează un fragment generat la fiecare N epoci - arată calitativ progresul."""

    def __init__(self, vocab: Vocabulary, seq_length: int, seed_text: str, every: int):
        super().__init__()
        self.vocab, self.seq_length, self.seed_text, self.every = vocab, seq_length, seed_text, every

    def on_epoch_end(self, epoch, logs=None):
        if self.every <= 0 or (epoch + 1) % self.every:
            return
        gen = CharGenerator(self.model, self.vocab, self.seq_length)
        res = gen.generate(self.seed_text, num_chars=200, temperature=0.5, rng_seed=0)
        print(f"\n--- Exemplu după epoca {epoch + 1} (T=0.5) ---\n{res.full_text}\n" + "-" * 50)


def train(cfg: TrainConfig) -> dict:
    keras.utils.set_random_seed(cfg.seed)  # reproductibilitate (Python, NumPy, TF)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Date ------------------------------------------------------------------
    text = load_corpus(cfg.data_path, cfg.unicode_normalization, cfg.max_chars)
    vocab = Vocabulary.from_text(text)
    corpus_stats = describe_corpus(text, vocab)
    encoded = vocab.encode(text)

    train_ids, val_ids = train_val_split(encoded, cfg.val_fraction, cfg.seq_length)
    step = cfg.resolved_step()
    train_ds, n_train = make_dataset(train_ids, cfg.seq_length, cfg.batch_size, step=step,
                                     shuffle=True, seed=cfg.seed, target_mode=cfg.target_mode)
    val_ds, n_val = make_dataset(val_ids, cfg.seq_length, cfg.batch_size, step=step,
                                 shuffle=False, target_mode=cfg.target_mode)
    targets_per_window = cfg.seq_length if cfg.target_mode == "many_to_many" else 1
    print(f"Mod target: {cfg.target_mode}, step={step}, "
          f"targeturi antrenate/epocă: {n_train * targets_per_window:,}")
    print(f"Caractere train / validation : {len(train_ids):,} / {len(val_ids):,}")
    print(f"Secvențe train / validation  : {n_train:,} / {n_val:,}  "
          f"(~{math.ceil(n_train / cfg.batch_size):,} pași/epocă)")

    # Verificare explicită a formelor tensorilor + a alinierii input/target.
    xb, yb = next(iter(train_ds))
    assert xb.shape[1] == cfg.seq_length and xb.shape[0] == yb.shape[0]
    print(f"Formă batch: x={tuple(xb.shape)} {xb.dtype.name}, y={tuple(yb.shape)} {yb.dtype.name}")
    print(f"Exemplu  input : {vocab.decode(xb[0].numpy()[-40:])!r} (ultimele 40 din {cfg.seq_length})")
    if cfg.target_mode == "many_to_many":
        assert tuple(yb.shape) == tuple(xb.shape)
        assert np.array_equal(xb[0, 1:].numpy(), yb[0, :-1].numpy())  # y = x deplasat cu 1
        print(f"Exemplu  target: {vocab.decode(yb[0].numpy()[-40:])!r} (x deplasat cu 1 poziție)")
    else:
        assert yb.shape.rank == 1
        print(f"Exemplu  target: {vocab.decode([int(yb[0])])!r}")

    # 2. Model -----------------------------------------------------------------
    model = build_model(vocab.size, cfg.embedding_dim, cfg.lstm_units, cfg.num_lstm_layers,
                        cfg.dropout, cfg.learning_rate, cfg.clipnorm, cfg.target_mode)
    model.summary()

    # Vocabularul se salvează ÎNAINTE de antrenare: e necesar pentru a folosi orice checkpoint.
    vocab.save(out_dir / VOCAB_FILE, seq_length=cfg.seq_length, target_mode=cfg.target_mode)
    (out_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                                               encoding="utf-8")

    # 3. Callbacks ---------------------------------------------------------------
    ckpt_path = out_dir / MODEL_FILE
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.reduce_lr_factor,
                                          patience=cfg.reduce_lr_patience, min_lr=cfg.min_lr, verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
        SampleTextCallback(vocab, cfg.seq_length, text[:cfg.seq_length], cfg.sample_every_n_epochs),
    ]

    # 4. Antrenare -------------------------------------------------------------
    t0 = time.time()
    # shuffle=False: amestecarea este deja făcută în tf.data (train_ds.shuffle).
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg.epochs,
                        callbacks=callbacks, shuffle=False)
    train_seconds = time.time() - t0
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")

    # 5. Reîncărcăm cel mai bun checkpoint de pe disc (verifică și salvarea/încărcarea)
    best_model = keras.models.load_model(ckpt_path)
    val_loss, val_acc, val_top5 = best_model.evaluate(val_ds, verbose=0)
    probs = best_model.predict(xb[:4], verbose=0)
    assert probs.shape[0] == min(4, xb.shape[0]) and probs.shape[-1] == vocab.size
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)  # softmax valid
    print(f"Model reîncărcat din {ckpt_path}: output {probs.shape}, sume probabilități ≈ 1 ✓")

    # 6. Evaluare ----------------------------------------------------------------
    uni_ppl = unigram_perplexity(train_ids, val_ids, vocab.size)
    fc_loss = full_context_loss(best_model, val_ids, cfg.seq_length)
    metrics = {
        **corpus_stats,
        "vocab_size": vocab.size,
        "seq_length": cfg.seq_length,
        "num_params": int(best_model.count_params()),
        "epochs_run": len(hist["loss"]),
        "train_seconds": round(train_seconds, 1),
        "final_train_loss": hist["loss"][-1],
        "final_train_accuracy": hist.get("accuracy", [None])[-1],
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_top5_accuracy": float(val_top5),
        "val_perplexity": perplexity(val_loss),
        "val_bits_per_char": bits_per_char(val_loss),
        "target_mode": cfg.target_mode,
        "full_context_loss": fc_loss,
        "full_context_perplexity": perplexity(fc_loss),
        "unigram_perplexity": uni_ppl,
        "uniform_perplexity": float(vocab.size),
    }
    metrics["diagnosis"] = diagnose(hist, uni_ppl)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(hist, out_dir)

    print("\n" + "=" * 70 + "\nEVALUARE (cel mai bun model, set de validare)\n" + "=" * 70)
    print(f"Validation loss       : {val_loss:.4f} nats/caracter")
    print(f"Validation accuracy   : {val_acc:.4f}   (top-5: {val_top5:.4f})")
    print(f"Perplexity            : {metrics['val_perplexity']:.3f}")
    print(f"Bits per character    : {metrics['val_bits_per_char']:.3f}")
    print(f"Perplexity (context complet de {cfg.seq_length} caractere): "
          f"{metrics['full_context_perplexity']:.3f}   <- comparabilă între moduri")
    print(f"Baseline uniform PP   : {vocab.size}  |  baseline unigram PP: {uni_ppl:.3f}")
    for m in metrics["diagnosis"]:
        print("-", m)
    print(f"\nArtefacte salvate în: {out_dir.resolve()}")
    return metrics


def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description="Antrenează generatorul de text LSTM.")
    for field, value in cfg.to_dict().items():
        flag = "--" + field.replace("_", "-")
        if field in ("max_chars", "step"):
            p.add_argument(flag, type=int, default=value)
        elif field == "target_mode":
            p.add_argument(flag, choices=["many_to_many", "many_to_one"], default=value)
        elif field == "unicode_normalization":
            p.add_argument(flag, choices=["NFC", "NFD", "NFKC", "NFKD"], default=value)
        elif field == "data_path":
            p.add_argument("--data", dest=field, default=value)
        else:
            p.add_argument(flag, type=type(value), default=value)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())
