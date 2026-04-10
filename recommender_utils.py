import ast
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
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
        raise FileNotFoundError(f"Dosya bulunamadi: {file_path}")

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


def build_numeric_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
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


def listened_artist_set(df: pd.DataFrame, listened_idx: Sequence[int]) -> set:
    artists = set()
    if "artists" not in df.columns:
        return artists
    for idx in listened_idx:
        artists.update(parse_listlike(df.loc[idx, "artists"]))
    return artists


def artist_overlap_score(df: pd.DataFrame, candidate_idx: Sequence[int], listened_artists: set) -> np.ndarray:
    if "artists" not in df.columns or not listened_artists:
        return np.zeros(len(candidate_idx), dtype=np.float32)

    scores = []
    for idx in candidate_idx:
        candidate_artists = set(parse_listlike(df.loc[idx, "artists"]))
        if not candidate_artists:
            scores.append(0.0)
            continue
        intersection = len(candidate_artists.intersection(listened_artists))
        union = len(candidate_artists.union(listened_artists))
        scores.append(intersection / union if union else 0.0)
    return np.asarray(scores, dtype=np.float32)


def build_result_frame(
    df: pd.DataFrame,
    candidate_idx: np.ndarray,
    top_k: int,
    score_columns: dict,
) -> pd.DataFrame:
    available_meta = [col for col in ["name", "artists", "album", "year"] if col in df.columns]
    result = df.loc[candidate_idx, available_meta].copy()
    result.insert(0, "candidate_index", candidate_idx)

    for col_name, values in score_columns.items():
        result[col_name] = values

    return result.sort_values("final_score", ascending=False).reset_index(drop=True).head(top_k)


def save_recommendations_table_image(
    recommendations: pd.DataFrame,
    out_png: str,
    title: str,
) -> None:
    display_df = recommendations.copy()
    for col in ["latent_similarity", "feature_similarity", "artist_similarity", "final_score", "embedding_similarity"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(6)

    fig_w = 24
    fig_h = max(6, 0.75 * (len(display_df) + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    plt.suptitle(title, fontsize=16, y=0.98)

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#E8F1F8")
            cell.set_text_props(weight="bold")
            cell.set_linewidth(0.8)

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_score_bar_chart(
    recommendations: pd.DataFrame,
    out_png: str,
    title: str,
    score_col: str = "final_score",
) -> None:
    if score_col not in recommendations.columns or "name" not in recommendations.columns:
        return

    chart_df = recommendations.copy().iloc[::-1]
    labels = [name if len(str(name)) <= 28 else f"{str(name)[:25]}..." for name in chart_df["name"]]
    scores = chart_df[score_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(labels, scores, color="#2A6F97")
    ax.set_title(title)
    ax.set_xlabel(score_col)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
