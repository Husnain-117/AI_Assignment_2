import os
import shutil
import argparse
import json
import numpy as np
from utils import preprocessing as prep
from utils.model_builder import build_model, make_callbacks
from utils.evaluation import plot_history, save_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/shakespeare_plays.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--vocab", type=int, default=15000)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=30)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--lstm", nargs="*", type=int, default=[256,256,128])
    ap.add_argument("--max_lines", type=int)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("reports/plots", exist_ok=True)

    data = prep.prepare(args.csv, args.out, vocab_size=args.vocab, min_len=args.min_len, max_len=args.max_len, step=args.step, val_ratio=args.val, max_lines=args.max_lines)
    Xtr, ytr, Xva, yva = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    cfg = data["config"]

    model = build_model(vocab_size=cfg["vocab_size"], max_seq_len=cfg["max_seq_len"], emb_dim=args.emb, lstm_units=args.lstm)
    cbs = make_callbacks(args.out)

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        batch_size=args.batch,
        epochs=args.epochs,
        callbacks=cbs,
        verbose=1,
    )

    model.save(os.path.join(args.out, "final_model.keras"))

    plot_history(os.path.join(args.out, "history.csv"), "reports/plots")
    # Duplicate plot for report's Question 2 expected filename
    src_plot = os.path.join("reports/plots", "training_curves.png")
    dst_plot = os.path.join("reports/plots", "q2_training_curves.png")
    if os.path.exists(src_plot):
        try:
            shutil.copyfile(src_plot, dst_plot)
        except Exception:
            pass
    save_summary(os.path.join(args.out, "history.csv"), os.path.join(args.out, "best_summary.json"))

    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "data_config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
