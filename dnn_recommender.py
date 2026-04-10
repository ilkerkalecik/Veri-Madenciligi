import argparse
import ast
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


EXPECTED_COLUMNS = [
    "id", "name", "album", "album_id", "artists", "artist_ids",
    "track_number", "disc_number", "explicit", "danceability", "energy",
    "key", "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "time_signature", "year", "release_date"
]

NUMERIC_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "time_signature", "year", "track_number", "disc_number"
]


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception:
        df = pd.read_csv(file_path)

    if len(df.columns) >= len(EXPECTED_COLUMNS):
        df = df.iloc[:, :len(EXPECTED_COLUMNS)].copy()
        df.columns = EXPECTED_COLUMNS

    df.columns = [str(col).strip().lower() for col in df.columns]
    return df.reset_index(drop=True)


def parse_listlike(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    except Exception:
        return [part.strip().strip("'").strip('"') for part in text.split(",") if part.strip()]


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    work_df = df.copy()

    available_numeric = [col for col in NUMERIC_FEATURES if col in work_df.columns]
    for col in available_numeric:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    if "explicit" in work_df.columns:
        work_df["explicit"] = (
            work_df["explicit"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(float)
        )
        available_numeric.append("explicit")

    numeric_frame = work_df[available_numeric].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    numeric_matrix = scaler.fit_transform(imputer.fit_transform(numeric_frame))

    return numeric_matrix.astype(np.float32), available_numeric


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def get_bottleneck_index(hidden_layer_sizes: Tuple[int, ...]) -> int:
    return int(np.argmin(hidden_layer_sizes))


def encode_with_trained_autoencoder(model: MLPRegressor, features: np.ndarray) -> np.ndarray:
    activations = features
    bottleneck_layer = get_bottleneck_index(model.hidden_layer_sizes)

    for layer_idx, (weights, bias) in enumerate(zip(model.coefs_[:-1], model.intercepts_[:-1])):
        activations = relu(np.matmul(activations, weights) + bias)
        if layer_idx == bottleneck_layer:
            return activations

    raise RuntimeError("Bottleneck katmanı hesaplanamadı.")


def choose_hidden_layers(n_features: int, n_samples: int) -> Tuple[int, ...]:
    wide = max(16, min(64, n_features * 2))
    mid = max(8, min(32, n_features))
    bottleneck = max(4, min(16, n_features // 2 if n_features > 1 else 4, max(4, n_samples // 20)))
    return (wide, mid, bottleneck, mid, wide)


def train_autoencoder(features: np.ndarray) -> MLPRegressor:
    hidden_layers = choose_hidden_layers(features.shape[1], features.shape[0])
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
        max_iter=120,
        random_state=42,
        verbose=False,
    )
    model.fit(features, features)
    return model


def recommend_next_songs(
    df: pd.DataFrame,
    features: np.ndarray,
    base_size: int,
    top_k: int,
) -> pd.DataFrame:
    if len(df) <= base_size:
        raise ValueError(f"Veri sayısı {len(df)}. En az {base_size + 1} şarkı olmalı.")

    model = train_autoencoder(features)
    embeddings = encode_with_trained_autoencoder(model, features)

    listened_idx = np.arange(base_size)
    candidate_idx = np.arange(base_size, len(df))

    user_latent_profile = embeddings[listened_idx].mean(axis=0, keepdims=True)
    user_raw_profile = features[listened_idx].mean(axis=0, keepdims=True)
    listened_artists = set()
    if "artists" in df.columns:
        for idx in listened_idx:
            listened_artists.update(parse_listlike(df.loc[idx, "artists"]))

    latent_scores = cosine_similarity(embeddings[candidate_idx], user_latent_profile).ravel()
    raw_scores = cosine_similarity(features[candidate_idx], user_raw_profile).ravel()
    artist_scores = []
    for idx in candidate_idx:
        candidate_artists = set(parse_listlike(df.loc[idx, "artists"])) if "artists" in df.columns else set()
        if not candidate_artists or not listened_artists:
            artist_scores.append(0.0)
            continue
        intersection = len(candidate_artists.intersection(listened_artists))
        union = len(candidate_artists.union(listened_artists))
        artist_scores.append(intersection / union if union else 0.0)
    artist_scores = np.asarray(artist_scores, dtype=np.float32)

    final_scores = 0.70 * latent_scores + 0.20 * raw_scores + 0.10 * artist_scores

    result = df.loc[candidate_idx, ["name", "artists", "album", "year"]].copy()
    result.insert(0, "candidate_index", candidate_idx)
    result["latent_similarity"] = latent_scores
    result["feature_similarity"] = raw_scores
    result["artist_similarity"] = artist_scores
    result["final_score"] = final_scores

    listened_names = df.loc[listened_idx, "name"].tolist() if "name" in df.columns else listened_idx.tolist()
    listened_preview = " | ".join(str(name) for name in listened_names[:5])
    result["profile_summary"] = f"Ilk {base_size} sarki profili: {listened_preview}"

    result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
    return result.head(top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description="DNN tabanli sarki onerici")
    parser.add_argument("--file", default="shakira.csv", help="CSV veya Excel veri dosyasi")
    parser.add_argument("--base-size", type=int, default=25, help="Kullanicinin dinledigi ilk sarki sayisi")
    parser.add_argument("--top-k", type=int, default=10, help="Kac sarki onerilecegi")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    file_path = args.file if os.path.isabs(args.file) else os.path.join(base_dir, args.file)

    try:
        df = load_dataset(file_path)
        features, _ = build_feature_matrix(df)
        recommendations = recommend_next_songs(df, features, base_size=args.base_size, top_k=args.top_k)
    except Exception as exc:
        print(f"HATA: {exc}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(base_dir, f"{base_name}_dnn_recommendations_base{args.base_size}.csv")
    recommendations.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Veri dosyasi: {file_path}")
    print(f"Dinleme gecmisi olarak kullanilan ilk sarki sayisi: {args.base_size}")
    print(f"Oneri sayisi: {args.top_k}")
    print(f"Sonuclar kaydedildi: {out_csv}\n")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
