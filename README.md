# Laboratorul 1 - Generator de text LSTM caracter-cu-caracter

Model LSTM antrenat de la zero (fără modele pre-antrenate) care învață să prezică următorul caracter dintr-o secvență și generează text nou, caracter-cu-caracter, în limba română.

## Structura proiectului

| Fișier | Ce face |
|---|---|
| `wikipedia.py` | Colectează textul brut (sursa corpusului). |
| `prep.py` | Curăță/pregătește textul brut și scrie `corpus_ro_clean.txt`. |
| `corpus_ro_clean.txt` | Corpusul final, curățat, folosit la antrenare. |
| `config.py` | Toți hiperparametrii și căile proiectului, într-un singur loc (`TrainConfig`). |
| `data_utils.py` | Încărcare corpus, vocabular caracter↔id, split train/validare fără leakage, construcție `tf.data.Dataset` (input, target). |
| `model.py` | Arhitectura LSTM (`build_model`): Embedding → LSTM → Dropout → LSTM → Dropout → Dense(softmax). |
| `train.py` | Pipeline complet: date → model → antrenare cu callbacks → salvare → evaluare. Script principal de rulat. |
| `evaluate.py` | Calculează perplexitate/BPC, generează graficele loss/accuracy/perplexity, diagnostic overfitting/underfitting. |
| `generate.py` | Funcția `generate_text(seed, num_chars, temperature)` + sampling cu temperatură; rulabil și din CLI. |
| `experiments.py` | Compară temperaturi (0.2/0.5/1.0/1.5) și, opțional, valori diferite de `seq_length`. |
| `app.py` | **Interfața Streamlit**: generare interactivă (seed, nr. caractere, temperatură), comparare temperaturi, tab de evaluare cu grafice. |
| `requirements.txt` | Dependențe (TensorFlow/Keras, NumPy, Matplotlib, Streamlit). |

## Arhitectura modelului
Input (batch, T) int32
→ Embedding(vocab_size, 128)
→ LSTM(256, return_sequences=True)
→ Dropout(0.2)
→ LSTM(256, return_sequences=True)
→ Dropout(0.2)
→ Dense(vocab_size, softmax)


Cu un vocabular de 32 de caractere: **931 872 parametri**.

## Hiperparametri (`config.py`)

| Parametru | Valoare | Justificare |
|---|---|---|
| `SEQ_LENGTH` | 100 | context suficient, standard pentru char-RNN, cost rezonabil la BPTT |
| `EMBEDDING_DIM` | 128 | suficient pentru un vocabular mic, fără redundanță excesivă |
| `LSTM_UNITS` × `NUM_LSTM_LAYERS` | 256 × 2 | capacitate potrivită pentru corpus ~740k caractere, fără memorare |
| `DROPOUT` | 0.2 | regularizare standard, previne overfitting |
| `LEARNING_RATE` | 0.001 (Adam) | valoare implicită robustă, ajustată automat prin `ReduceLROnPlateau` |
| `CLIPNORM` | 1.0 | previne exploding gradients pe secvențe lungi |
| `BATCH_SIZE` | 128 | compromis stabilitate gradient / memorie |

## Funcția de loss

`sparse_categorical_crossentropy`:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)$$

Target = id întreg (nu one-hot). În modul `many_to_many`, se calculează la fiecare din cele 100 de poziții ale ferestrei și se mediază — echivalent cu maximum likelihood estimation.

## Monitorizarea antrenării

- `training_log.csv` — loss, accuracy, top5_accuracy, learning_rate, per epocă (train + validare)
- `ModelCheckpoint` — salvează `model.keras` doar la scăderea `val_loss`
- `EarlyStopping` (patience=5) — oprește și restaurează cele mai bune ponderi
- `ReduceLROnPlateau` (patience=2, factor=0.5) — reduce learning rate la stagnare
- eșantion de text generat la fiecare N epoci (verificare calitativă)

## Evaluarea calității generării

Dincolo de loss/accuracy, `experiments.py` calculează pe textul **generat**:

- **entropie medie** (biți) — predictibilitate
- **proporția de 4-grame distincte** — detectează repetitivitatea/buclele
- **proporția de cuvinte existente în corpus** — cât de mult "inventează" modelul
- **perplexitate pe context complet** — comparabilă corect între configurații
- comparație cu baseline uniform (= `vocab_size`) și unigram (frecvența caracterelor)

## Interpretarea rezultatelor (exemplu, epoca 15)

`val_loss = 1.629` → perplexitate ≈ 5,1 (vs. baseline uniform 32) — modelul folosește contextul, nu doar frecvențele caracterelor. `val_accuracy ≈ 50%`, `top5_accuracy ≈ 83%` — distribuție bine calibrată. Train ≈ validare → fără overfitting.

## Rulare

```bash
pip install -r requirements.txt

python train.py                 # antrenare
python generate.py --seed "text" --num-chars 300 --temperature 0.5
python experiments.py temperature --seed "text"
streamlit run app.py             # interfața
```
