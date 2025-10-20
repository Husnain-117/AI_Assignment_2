import os
import re
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

_ws = re.compile(r"\s+")
_brackets = re.compile(r"\[(?:[^\]]+)\]")
_nonalpha = re.compile(r"[^a-zA-Z\s\.,;:'!?-]")


def _clean(s: str) -> str:
    s = s or ""
    s = _brackets.sub(" ", s)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.lower()
    s = _nonalpha.sub(" ", s)
    s = _ws.sub(" ", s).strip()
    return s


def load_csv(csv_path: str, text_cols: List[str] = None, max_lines: int | None = None) -> List[str]:
    if text_cols is None:
        text_cols = ["PlayerLine", "player_line", "line", "text", "Dialogue", "dialogue"]
    df = pd.read_csv(csv_path)
    col = next((c for c in text_cols if c in df.columns), None)
    if not col:
        raise ValueError(f"No text column found in {csv_path}. Columns: {list(df.columns)}")
    lines = df[col].astype(str).tolist()
    if max_lines:
        lines = lines[:max_lines]
    return [x for x in lines if isinstance(x, str) and x.strip()]


def clean_corpus(lines: List[str]) -> List[str]:
    out = []
    for l in lines:
        c = _clean(l)
        if c:
            out.append(c)
    return out


def build_tokenizer(texts: List[str], vocab_size: int = 15000) -> Tokenizer:
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters="")
    tok.fit_on_texts(texts)
    return tok


def sequences_from_texts(texts: List[str], tok: Tokenizer, min_len: int = 3, max_len: int = 30, step: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    seqs = tok.texts_to_sequences(texts)
    X, y = [], []
    max_in_len = 0
    for s in seqs:
        if len(s) < min_len:
            continue
        for end in range(min_len, min(len(s), max_len)+1):
            start = 0
            while start + end <= len(s):
                in_seq = s[start:start+end-1]
                out_tok = s[start+end-1]
                if len(in_seq) >= min_len-1:
                    X.append(in_seq)
                    y.append(out_tok)
                    max_in_len = max(max_in_len, len(in_seq))
                start += step
    X = pad_sequences(X, maxlen=max_in_len, padding="pre")
    y = np.array(y)
    return X, y, max_in_len


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X)*(1-val_ratio))
    ti, vi = idx[:cut], idx[cut:]
    return X[ti], X[vi], y[ti], y[vi]


def save_artifacts(out_dir: str, tok: Tokenizer, cfg: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(tok, os.path.join(out_dir, "tokenizer.pickle"))
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def prepare(csv_path: str, out_dir: str, vocab_size: int = 15000, min_len: int = 3, max_len: int = 30, step: int = 1, val_ratio: float = 0.2, max_lines: int | None = None) -> Dict[str, Any]:
    lines = load_csv(csv_path, max_lines=max_lines)
    texts = clean_corpus(lines)
    tok = build_tokenizer(texts, vocab_size)
    X, y, max_in_len = sequences_from_texts(texts, tok, min_len, max_len, step)
    Xtr, Xva, ytr, yva = split_train_val(X, y, val_ratio)
    cfg = {"vocab_size": min(vocab_size, len(tok.word_index)+1), "max_seq_len": int(X.shape[1]), "min_len": min_len, "max_len": max_len, "step": step, "val_ratio": val_ratio}
    save_artifacts(out_dir, tok, cfg)
    np.save(os.path.join(out_dir, "X_train.npy"), Xtr)
    np.save(os.path.join(out_dir, "y_train.npy"), ytr)
    np.save(os.path.join(out_dir, "X_val.npy"), Xva)
    np.save(os.path.join(out_dir, "y_val.npy"), yva)
    return {"tokenizer": tok, "config": cfg, "X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva}


def load_prepared(out_dir: str) -> Dict[str, Any]:
    tok = joblib.load(os.path.join(out_dir, "tokenizer.pickle"))
    with open(os.path.join(out_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    Xtr = np.load(os.path.join(out_dir, "X_train.npy"))
    ytr = np.load(os.path.join(out_dir, "y_train.npy"))
    Xva = np.load(os.path.join(out_dir, "X_val.npy"))
    yva = np.load(os.path.join(out_dir, "y_val.npy"))
    return {"tokenizer": tok, "config": cfg, "X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva}
