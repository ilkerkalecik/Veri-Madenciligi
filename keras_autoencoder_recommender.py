import argparse
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from recommender_utils import (
    artist_overlap_score,
    build_numeric_feature_matrix,
    build_result_frame,
    listened_artist_set,
    load_dataset,
    save_recommendations_table_image,
    save_score_bar_chart,
)

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as exc:
    print("HATA: TensorFlow/Keras import edilemedi.")
    print("Kurulum: python3 -m pip install tensorflow")
    print(f"Detay: {exc}")
    sys.exit(1)


def build_autoencoder(input_dim: int, latent_dim: int) -> tuple:
    inputs = keras.Input(shape=(input_dim,), name="song_features")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dense(32, activation="relu")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent_vector")(x)
    x = keras.layers.Dense(32, activation="relu")(latent)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="song_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="song_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return autoencoder, encoder


def recommend_with_autoencoder(features: np.ndarray, latent_dim: int, epochs: int, batch_size: int):
    tf.random.set_seed(42)
    np.random.seed(42)

    autoencoder, encoder = build_autoencoder(features.shape[1], latent_dim)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )
    history = autoencoder.fit(
        features,
        features,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stop],
    )
    embeddings = encoder.predict(features, verbose=0)
    return embeddings, history


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorFlow/Keras autoencoder tabanli sarki onerici")
    parser.add_argument("--file", default="shakira.csv", help="CSV veya Excel veri dosyasi")
    parser.add_argument("--base-size", type=int, default=25, help="Ilk kac sarki dinlenmis kabul edilsin")
    parser.add_argument("--top-k", type=int, default=10, help="Kac sarki onerilsin")
    parser.add_argument("--latent-dim", type=int, default=12, help="Latent temsil boyutu")
    parser.add_argument("--epochs", type=int, default=60, help="Maksimum epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch boyutu")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    file_path = args.file if os.path.isabs(args.file) else os.path.join(base_dir, args.file)

    try:
        df = load_dataset(file_path)
        if len(df) <= args.base_size:
            raise ValueError(f"Veri sayisi {len(df)}. En az {args.base_size + 1} sarki olmali.")

        features, _ = build_numeric_feature_matrix(df)
        embeddings, history = recommend_with_autoencoder(
            features=features,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"HATA: {exc}")
        sys.exit(1)

    listened_idx = np.arange(args.base_size)
    candidate_idx = np.arange(args.base_size, len(df))

    user_latent_profile = embeddings[listened_idx].mean(axis=0, keepdims=True)
    user_feature_profile = features[listened_idx].mean(axis=0, keepdims=True)
    listened_artists = listened_artist_set(df, listened_idx)

    latent_scores = cosine_similarity(embeddings[candidate_idx], user_latent_profile).ravel()
    feature_scores = cosine_similarity(features[candidate_idx], user_feature_profile).ravel()
    artist_scores = artist_overlap_score(df, candidate_idx, listened_artists)
    final_scores = 0.75 * latent_scores + 0.15 * feature_scores + 0.10 * artist_scores

    recommendations = build_result_frame(
        df=df,
        candidate_idx=candidate_idx,
        top_k=args.top_k,
        score_columns={
            "latent_similarity": latent_scores,
            "feature_similarity": feature_scores,
            "artist_similarity": artist_scores,
            "final_score": final_scores,
        },
    )

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(base_dir, f"{base_name}_keras_autoencoder_base{args.base_size}.csv")
    out_table_png = os.path.join(base_dir, f"{base_name}_keras_autoencoder_base{args.base_size}_table.png")
    out_chart_png = os.path.join(base_dir, f"{base_name}_keras_autoencoder_base{args.base_size}_scores.png")
    recommendations.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_recommendations_table_image(
        recommendations=recommendations,
        out_png=out_table_png,
        title=f"Keras Autoencoder Top {args.top_k} Oneri Tablosu",
    )
    save_score_bar_chart(
        recommendations=recommendations,
        out_png=out_chart_png,
        title=f"Keras Autoencoder Top {args.top_k} Final Score",
        score_col="final_score",
    )

    best_val_loss = min(history.history["val_loss"]) if history.history.get("val_loss") else None
    print(f"Veri dosyasi: {file_path}")
    print(f"Dinlenmis ilk sarki sayisi: {args.base_size}")
    print(f"Oneri sayisi: {args.top_k}")
    print(f"Kayit dosyasi: {out_csv}")
    print(f"Tablo gorseli: {out_table_png}")
    print(f"Skor grafigi: {out_chart_png}")
    if best_val_loss is not None:
        print(f"En iyi validation loss: {best_val_loss:.6f}")
    print()
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
