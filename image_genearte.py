# scripts/generate_q2_figs.py
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
# Script is at project root; ensure PROJ points to current folder (not parent)
PROJ = ROOT
MODELS_DIR = os.path.join(PROJ, "models")
PLOTS_DIR = os.path.join(PROJ, "reports", "plots")

# Fallback: if models/ not found here, search upwards
if not os.path.isdir(MODELS_DIR):
    cur = ROOT
    for _ in range(3):
        cand = os.path.join(cur, "models")
        if os.path.isdir(cand):
            MODELS_DIR = cand
            PLOTS_DIR = os.path.join(cur, "reports", "plots")
            break
        cur = os.path.dirname(cur)

BEST_MODEL = os.path.join(MODELS_DIR, "best_model.keras")
FINAL_MODEL = os.path.join(MODELS_DIR, "final_model.keras")
TOK_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")
CFG_PATH = os.path.join(MODELS_DIR, "config.json")

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_artifacts():
    if not os.path.exists(TOK_PATH) or not os.path.exists(CFG_PATH):
        raise FileNotFoundError("Missing tokenizer/config. Train first.")
    tok = joblib.load(TOK_PATH)
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_path = BEST_MODEL if os.path.exists(BEST_MODEL) else FINAL_MODEL
    if not os.path.exists(model_path):
        raise FileNotFoundError("Missing best_model.keras/final_model.keras. Train first.")
    model = keras.models.load_model(model_path)
    return tok, cfg, model

def apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-9, 1.0)) / max(1e-3, temp)
    exp = np.exp(logits)
    return exp / np.sum(exp)

def topk_predict(tok, cfg, model, text: str, k: int = 5, temp: float = 1.0):
    seq = tok.texts_to_sequences([text.lower().strip()])
    seq = pad_sequences(seq, maxlen=cfg["max_seq_len"], padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    probs = apply_temperature(probs, temp)
    top_idx = probs.argsort()[-k:][::-1]
    inv = {v: w for w, v in tok.word_index.items()}
    words = [inv.get(int(i), "<OOV>") for i in top_idx]
    return list(zip(words, probs[top_idx].tolist()))

def figure_q2_examples(save_path: str, prompts: list[str], tok, cfg, model, k: int = 5, temp: float = 1.0):
    rows = len(prompts)
    fig, axs = plt.subplots(rows, 1, figsize=(10, 2.2 * rows))
    if rows == 1:
        axs = [axs]
    for ax, prompt in zip(axs, prompts):
        preds = topk_predict(tok, cfg, model, prompt, k=k, temp=temp)
        words = [w for w, _ in preds]
        probs = [p for _, p in preds]
        ax.barh(range(len(words))[::-1], probs[::-1], color="#6baed6")
        ax.set_yticks(range(len(words))[::-1], labels=words[::-1])
        ax.set_xlabel("Probability")
        ax.set_title(f"Prompt: {prompt}")
        ax.set_xlim(0, max(0.001, max(probs) * 1.1))
        ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

def figure_q2_compare(save_path: str, histories: list[str]):
    """
    histories: list of paths to CSV logs (e.g., ["models/history.csv", "models/exp_deep/history.csv"])
    If only one exists, still plot a single bar.
    """
    data = []
    for h in histories:
        if os.path.exists(h):
            df = pd.read_csv(h)
            # best by val_accuracy if available; fallback to last
            row = df.loc[df["val_accuracy"].idxmax()] if "val_accuracy" in df.columns else df.iloc[-1]
            name = os.path.basename(os.path.dirname(os.path.abspath(h))) or "models"
            data.append({
                "name": name,
                "val_acc": float(row.get("val_accuracy", float("nan"))),
                "val_loss": float(row.get("val_loss", float("nan"))),
                "perplexity": perplexity_from_loss(float(row.get("val_loss", float("nan")))) if "val_loss" in row else float("nan"),
            })
    if not data:
        print("No history.csv files found; skipping comparison plot.")
        return

    names = [d["name"] for d in data]
    accs = [d["val_acc"] for d in data]
    ppls = [d["perplexity"] for d in data]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    w = 0.35

    b1 = ax1.bar(x - w/2, accs, width=w, color="#74c476", label="Val Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Val Accuracy")
    ax1.set_xticks(x, names, rotation=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + w/2, ppls, width=w, color="#6baed6", label="Perplexity")
    ax2.set_ylabel("Perplexity")

    lines = [b1, b2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.suptitle("Question 2: Validation Accuracy and Perplexity Comparison")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def main():
    tok, cfg, model = load_artifacts()

    # 1) Examples figure
    prompts = [
        "To be or not to",
        "O Romeo, Romeo wherefore",
        "If music be the",
        "Thus with a kiss I",
        "Friends, Romans, countrymen,"
    ]
    examples_path = os.path.join(PLOTS_DIR, "q2_examples.png")
    figure_q2_examples(examples_path, prompts, tok, cfg, model, k=5, temp=1.0)
    print(f"Saved: {examples_path}")

    # 2) Comparison figure (uses available history logs)
    # Start with the default history; add more paths if you run more experiments
    histories = [
        os.path.join(MODELS_DIR, "history.csv"),
        # Example of additional runs (uncomment and adjust if you save experiments under subfolders)
        # os.path.join(MODELS_DIR, "exp_deep", "history.csv"),
        # os.path.join(MODELS_DIR, "exp_wide", "history.csv"),
        # os.path.join(MODELS_DIR, "exp_reg", "history.csv"),
    ]
    compare_path = os.path.join(PLOTS_DIR, "q2_compare.png")
    figure_q2_compare(compare_path, histories)
    print(f"Saved: {compare_path}")

if __name__ == "__main__":
    main()
