from typing import Dict, Any
from tensorflow import keras
from tensorflow.keras import layers


def build_model(vocab_size: int, max_seq_len: int, emb_dim: int = 256,
                lstm_units: list[int] = [256, 256, 128], dropout: float = 0.2,
                recurrent_dropout: float = 0.0) -> keras.Model:
    inputs = keras.Input(shape=(max_seq_len,), dtype="int32")
    x = layers.Embedding(vocab_size, emb_dim, input_length=max_seq_len, mask_zero=True)(inputs)
    for i, u in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = layers.LSTM(u, return_sequences=return_seq, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipvalue=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
        ],
    )
    return model


def make_callbacks(out_dir: str) -> list[Any]:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=f"{out_dir}/best_model.keras", monitor="val_accuracy", save_best_only=True),
        keras.callbacks.CSVLogger(f"{out_dir}/history.csv", append=False),
    ]
