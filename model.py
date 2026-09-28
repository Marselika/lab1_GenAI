"""
model.py - Arhitectura LSTM pentru predicția următorului caracter.

    Input (batch, T) int32
      -> Embedding          (batch, T, EMBEDDING_DIM)
      -> LSTM #1            (batch, T, LSTM_UNITS)   return_sequences=True
      -> Dropout
      -> LSTM #2            (batch, LSTM_UNITS)      many_to_one: doar ultima stare
                            (batch, T, LSTM_UNITS)   many_to_many: starea la fiecare pas
      -> Dropout
      -> Dense + softmax    (batch, VOCAB_SIZE)      many_to_one
                            (batch, T, VOCAB_SIZE)   many_to_many: P(c[t+1] | c[0..t]) la fiecare t

T = None: modelul acceptă secvențe de orice lungime. Antrenăm pe T = SEQ_LENGTH,
dar la generare seed-ul poate fi mai scurt de SEQ_LENGTH caractere.
"""
from __future__ import annotations

import keras
from keras import layers


def build_model(vocab_size: int,
                embedding_dim: int = 128,
                lstm_units: int = 256,
                num_lstm_layers: int = 2,
                dropout: float = 0.2,
                learning_rate: float = 1e-3,
                clipnorm: float | None = 1.0,
                target_mode: str = "many_to_many") -> keras.Model:
    if num_lstm_layers < 1:
        raise ValueError("num_lstm_layers trebuie să fie >= 1")

    inputs = keras.Input(shape=(None,), dtype="int32", name="char_ids")

    # Embedding: fiecare id de caracter -> vector dens, învățat.
    x = layers.Embedding(vocab_size, embedding_dim, name="embedding")(inputs)

    for i in range(num_lstm_layers):
        is_last = i == num_lstm_layers - 1
        # Straturile intermediare returnează mereu toată secvența (stratul următor
        # are nevoie de ea). Ultimul strat: many_to_one -> doar starea finală;
        # many_to_many -> starea de la fiecare pas (o predicție per poziție).
        return_seq = (not is_last) or target_mode == "many_to_many"
        # NU folosim recurrent_dropout: ar dezactiva kernelul rapid cuDNN pe GPU.
        x = layers.LSTM(lstm_units, return_sequences=return_seq, name=f"lstm_{i + 1}")(x)
        x = layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)

    # Distribuția de probabilitate peste tot vocabularul. Pe un tensor 3D, Dense
    # se aplică independent la fiecare pas de timp: (B, T, units) -> (B, T, V).
    outputs = layers.Dense(vocab_size, activation="softmax", name="next_char_probs")(x)

    model = keras.Model(inputs, outputs, name="char_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        # Target = id întreg (nu one-hot) -> sparse categorical cross-entropy.
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )
    return model
