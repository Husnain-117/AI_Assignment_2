import os
import math
import json
from typing import Dict, Any
import matplotlib.pyplot as plt
import pandas as pd


def perplexity_from_loss(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def plot_history(csv_path: str, out_dir: str):
    if not os.path.exists(csv_path):
        return
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(len(df)))
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    axs[0].plot(df["epoch"], df["accuracy"], label="train")
    axs[0].plot(df["epoch"], df["val_accuracy"], label="val")
    axs[0].set_title("Accuracy")
    axs[0].legend()
    axs[1].plot(df["epoch"], df["loss"], label="train")
    axs[1].plot(df["epoch"], df["val_loss"], label="val")
    axs[1].set_title("Loss")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close(fig)


def save_summary(history_csv: str, out_json: str):
    if not os.path.exists(history_csv):
        return
    df = pd.read_csv(history_csv)
    best = df.loc[df["val_accuracy"].idxmax()].to_dict()
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
