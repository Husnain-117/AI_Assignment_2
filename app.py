import os
import json
import numpy as np
import streamlit as st
from tensorflow import keras
import joblib

ART_DIR = "models"
TOK_PATH = os.path.join(ART_DIR, "tokenizer.pickle")
CFG_PATH = os.path.join(ART_DIR, "config.json")
BEST_MODEL = os.path.join(ART_DIR, "best_model.keras")
FALLBACK_MODEL = os.path.join(ART_DIR, "final_model.keras")

@st.cache_resource
def load_artifacts():
    tok = joblib.load(TOK_PATH)
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_path = BEST_MODEL if os.path.exists(BEST_MODEL) else FALLBACK_MODEL
    model = keras.models.load_model(model_path)
    return tok, cfg, model


def apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-9, 1)) / max(1e-3, temp)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def predict_next(tok, cfg, model, text: str, k: int = 5, temp: float = 1.0):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tok.texts_to_sequences([text.lower().strip()])
    seq = pad_sequences(seq, maxlen=cfg["max_seq_len"], padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    probs = apply_temperature(probs, temp)
    top_idx = probs.argsort()[-k:][::-1]
    inv = {v:k for k,v in tok.word_index.items()}
    words = [inv.get(i, "<OOV>") for i in top_idx]
    return list(zip(words, probs[top_idx].tolist()))


def main():
    st.title("Shakespeare LSTM Sentence Completion")
    # Ensure artifacts exist before attempting to load
    if not os.path.exists(TOK_PATH) or not os.path.exists(CFG_PATH) or (
        not os.path.exists(BEST_MODEL) and not os.path.exists(FALLBACK_MODEL)
    ):
        st.warning("Model artifacts not found in `models/`. Please train the model first.")
        st.info(
            "Open a terminal and run:\n\n"
            "`python train.py --csv data/shakespeare_plays.csv --epochs 3 --max_lines 5000` (smoke test)\n\n"
            "After training finishes, restart this app."
        )
        return

    tok, cfg, model = load_artifacts()

    st.sidebar.header("Settings")
    temp = st.sidebar.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
    topk = st.sidebar.slider("Top-K", 1, 10, 5, 1)

    prompt = st.text_area("Enter partial sentence", "To be or not to")
    col1, col2 = st.columns(2)
    if col1.button("Predict"):
        preds = predict_next(tok, cfg, model, prompt, k=topk, temp=temp)
        st.subheader("Suggestions")
        for w, p in preds:
            st.write(f"{w} — {p:.3f}")
    if col2.button("Clear"):
        st.experimental_rerun()

    if os.path.exists(os.path.join(ART_DIR, "history.csv")):
        st.sidebar.download_button("Download training history", data=open(os.path.join(ART_DIR, "history.csv"), "rb"), file_name="history.csv")

if __name__ == "__main__":
    main()
