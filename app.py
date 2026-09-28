"""
app.py - Interfață Streamlit pentru generatorul de text LSTM.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import ARTIFACTS_DIR, DEFAULT_TEMPERATURES
from data_utils import printable_char
from experiments import compare_temperatures
from generate import CharGenerator

st.set_page_config(page_title="Generator de text LSTM", page_icon="🔤", layout="wide")


@st.cache_resource(show_spinner="Se încarcă modelul...")
def load_generator(model_dir: str) -> CharGenerator:
    # cache_resource: modelul se încarcă O SINGURĂ DATĂ, nu la fiecare interacțiune.
    return CharGenerator.from_dir(model_dir)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


model_dir = Path(ARTIFACTS_DIR)
try:
    gen = load_generator(str(model_dir))
except FileNotFoundError as e:
    st.error(f"{e}\n\nAntrenează mai întâi modelul cu `python train.py`.")
    st.stop()

metrics = read_json(model_dir / "metrics.json") or {}

# ---------------------------------------------------------------------------
# Sidebar: informații despre model
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    st.metric("Dimensiune vocabular", gen.vocab.size)
    st.metric("Sequence length", gen.seq_length)
    st.metric("Număr de parametri", f"{gen.model.count_params():,}")
    if "val_perplexity" in metrics:
        st.metric("Perplexity (validare)", f"{metrics['val_perplexity']:.3f}")
    with st.expander("Vocabular"):
        st.write("  ".join(f"`{printable_char(c)}`" for c in gen.vocab.chars))
    with st.expander("Arhitectură"):
        lines = []
        gen.model.summary(print_fn=lambda s, **_: lines.append(s))
        st.code("\n".join(lines), language=None)

st.title("Generare de text caracter-cu-caracter (LSTM)")
tab_gen, tab_temp, tab_eval = st.tabs(["Generare", "Comparare temperaturi", "Evaluare"])

# ---------------------------------------------------------------------------
# Tab 1: generare
# ---------------------------------------------------------------------------
with tab_gen:
    seed_text = st.text_area("Text inițial (seed)", value="România este", height=100)
    c1, c2, c3 = st.columns(3)
    num_chars = c1.number_input("Număr de caractere generate", 1, 3000, 400, step=50)
    temperature = c2.slider("Temperatură", 0.05, 2.0, 0.5, 0.05)
    fixed = c3.checkbox("Rezultat reproductibil (seed aleator fix)", value=False)
    rng_seed = c3.number_input("Seed aleator", 0, 10_000, 42, disabled=not fixed)

    if st.button("Generate Text", type="primary"):
        try:
            with st.spinner("Se generează..."):
                res = gen.generate(seed_text, int(num_chars), float(temperature),
                                   int(rng_seed) if fixed else None)
            st.session_state["last"] = (res, float(temperature), int(num_chars))
        except ValueError as e:
            st.error(str(e))

    if "last" in st.session_state:
        res, t_used, n_used = st.session_state["last"]
        if res.unknown_chars:
            st.warning("Caractere ignorate (nu există în vocabular): "
                       + " ".join(repr(c) for c in res.unknown_chars))
        st.subheader("Text generat")
        st.markdown(f"**Seed:** `{res.seed_used[-200:]}`")
        st.text_area("Rezultat (seed + text nou)", res.full_text, height=300)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temperatura utilizată", t_used)
        m2.metric("Caractere generate", n_used)
        m3.metric("Entropie medie", f"{res.mean_entropy_bits:.2f} biți")
        m4.metric("Sequence length", gen.seq_length)

# ---------------------------------------------------------------------------
# Tab 2: comparare temperaturi
# ---------------------------------------------------------------------------
with tab_temp:
    st.write("Același seed și aceeași sursă de aleatorism; se schimbă doar temperatura.")
    seed_cmp = st.text_input("Seed pentru comparație", value="România este")
    n_cmp = st.number_input("Caractere per temperatură", 50, 1000, 250, step=50)
    temps = st.multiselect("Temperaturi", [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0],
                           default=list(DEFAULT_TEMPERATURES))
    if st.button("Compară temperaturile"):
        try:
            with st.spinner("Se generează pentru fiecare temperatură..."):
                st.session_state["cmp"] = compare_temperatures(gen, seed_cmp, int(n_cmp), sorted(temps))
        except ValueError as e:
            st.error(str(e))
    for r in st.session_state.get("cmp", []):
        st.markdown(f"#### T = {r['temperature']}  ·  entropie medie {r['mean_entropy_bits']:.2f} biți  ·  "
                    f"4-grame distincte {r['distinct_4gram_ratio']:.0%}")
        st.text_area(f"T={r['temperature']}", r["text"], height=150, label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Tab 3: evaluare
# ---------------------------------------------------------------------------
with tab_eval:
    if not metrics:
        st.info("metrics.json lipsește - rulează train.py.")
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Train loss (final)", f"{metrics['final_train_loss']:.4f}")
        e2.metric("Validation loss", f"{metrics['val_loss']:.4f}")
        e3.metric("Validation accuracy", f"{metrics['val_accuracy']:.2%}")
        e4.metric("Perplexity", f"{metrics['val_perplexity']:.3f}")
        st.caption(f"Baseline-uri: uniform = {metrics['uniform_perplexity']:.0f}, "
                   f"unigram = {metrics['unigram_perplexity']:.2f}. "
                   f"Bits per character = {metrics['val_bits_per_char']:.3f}.")
        st.subheader("Diagnostic")
        for m in metrics.get("diagnosis", []):
            st.write("- " + m)
        for name in ("loss.png", "accuracy.png", "perplexity.png"):
            if (model_dir / name).exists():
                st.image(str(model_dir / name))
