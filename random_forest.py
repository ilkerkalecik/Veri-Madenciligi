import os
import sys
import time
import ast
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# İLERLEME ÇUBUĞU (opsiyonel)
# ---------------------------------------------------------
class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=30, fill='█', enabled=True):
        self.enabled = enabled
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0
        self.start_time = time.time()

    def update(self, progress=1):
        if not self.enabled:
            return
        self.iteration += progress
        self.iteration = min(self.iteration, self.total)

        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)

        elapsed = time.time() - self.start_time
        if self.iteration > 0:
            est_total = elapsed / (self.iteration / self.total)
            remaining = max(est_total - elapsed, 0)
            rem_str = f"{int(remaining)}sn kaldı"
        else:
            rem_str = "Hesaplanıyor..."

        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} [{rem_str}]')
        if self.iteration == self.total:
            sys.stdout.write('\n')
        sys.stdout.flush()


# ---------------------------------------------------------
# 1) VERİ OKUMA
# ---------------------------------------------------------
def load_data_forced_header(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"KRİTİK HATA: Dosya bulunamadı -> {file_path}")
        sys.exit(1)

    print(f"Dosya okunuyor: {file_path}")
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
        except:
            print("Dosya okunamadı.")
            sys.exit(1)

    if len(df.columns) < 2:
        print("UYARI: Veriler ayrıştırılıyor...")
        df = df.iloc[:, 0].astype(str).str.split(',', expand=True)

    correct_columns = [
        "id", "name", "album", "album_id", "artists", "artist_ids",
        "track_number", "disc_number", "explicit", "danceability", "energy",
        "key", "loudness", "mode", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
        "time_signature", "year", "release_date"
    ]

    if len(df.columns) >= len(correct_columns):
        df = df.iloc[:, :24]
        df.columns = correct_columns

    if 'danceability' in df.columns and str(df.iloc[0]['danceability']).strip().lower() == 'danceability':
        df = df.iloc[1:]

    df.columns = df.columns.str.strip().str.lower()
    return df.reset_index(drop=True)


# ---------------------------------------------------------
# 2) RULE-BASED SIMILARITY (candidate x history) - FAST
# ---------------------------------------------------------
AUDIO_01_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence"
]

NUMERIC_COLS_RULE = AUDIO_01_COLS + [
    "key", "loudness", "mode", "tempo", "duration_ms", "time_signature", "year"
]

def safe_to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def parse_listlike(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x]
    sx = str(x).strip()
    if sx == "" or sx.lower() == "nan":
        return []
    try:
        val = ast.literal_eval(sx)
        if isinstance(val, list):
            return [str(i).strip() for i in val]
        return [str(val).strip()]
    except:
        if "," in sx:
            return [p.strip().strip("'").strip('"') for p in sx.split(",") if p.strip()]
        return [sx.strip().strip("'").strip('"')]

def circular_key_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 12 - d)

def _w(weights: Optional[Dict[str, float]], name: str) -> float:
    if not weights:
        return 1.0
    return float(weights.get(name, 1.0))

def compute_similarity_sums_candidate_history(
    df: pd.DataFrame,
    candidates: List[int],
    history: List[int],
    rules: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Her candidate için: sum(edge_weight(candidate, history_j))
    Edge weight = tutan kural ağırlık toplamı (int)
    Komple n×n yok -> sadece (Nc × Nh) hesaplar.
    """
    cand = df.loc[candidates].copy()
    hist = df.loc[history].copy()

    # numeric parse
    for c in NUMERIC_COLS_RULE:
        if c in cand.columns:
            cand[c] = cand[c].apply(safe_to_float)
        if c in hist.columns:
            hist[c] = hist[c].apply(safe_to_float)

    # explicit -> 0/1
    if "explicit" in cand.columns:
        cand["explicit"] = cand["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)
    if "explicit" in hist.columns:
        hist["explicit"] = hist["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    Nc = len(candidates)
    Nh = len(history)
    score = np.zeros((Nc, Nh), dtype=np.float32)

    # --- audio 0-1 ---
    audio_eps = float(rules.get("audio_eps", 0.10))
    for c in AUDIO_01_COLS:
        if c not in cand.columns or c not in hist.columns:
            continue
        a = cand[c].to_numpy(dtype=np.float32)[:, None]          # (Nc,1)
        b = hist[c].to_numpy(dtype=np.float32)[None, :]          # (1,Nh)
        diff = np.abs(a - b)
        ok = (diff <= audio_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, c)

    # --- key circular ---
    if "key" in cand.columns and "key" in hist.columns:
        key_eps = float(rules.get("key_eps", 1.0))
        a = cand["key"].to_numpy(dtype=np.float32)[:, None]
        b = hist["key"].to_numpy(dtype=np.float32)[None, :]
        d = circular_key_distance(a, b)
        ok = (d <= key_eps) & np.isfinite(d)
        score += ok.astype(np.float32) * _w(weights, "key")

    # --- mode exact ---
    if "mode" in cand.columns and "mode" in hist.columns:
        a = cand["mode"].to_numpy(dtype=np.float32)[:, None]
        b = hist["mode"].to_numpy(dtype=np.float32)[None, :]
        ok = (a == b) & np.isfinite(a) & np.isfinite(b)
        score += ok.astype(np.float32) * _w(weights, "mode")

    # --- loudness ---
    if "loudness" in cand.columns and "loudness" in hist.columns:
        loud_eps = float(rules.get("loudness_eps", 2.0))
        a = cand["loudness"].to_numpy(dtype=np.float32)[:, None]
        b = hist["loudness"].to_numpy(dtype=np.float32)[None, :]
        diff = np.abs(a - b)
        ok = (diff <= loud_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "loudness")

    # --- tempo ---
    if "tempo" in cand.columns and "tempo" in hist.columns:
        tempo_eps = float(rules.get("tempo_eps", 6.0))
        a = cand["tempo"].to_numpy(dtype=np.float32)[:, None]
        b = hist["tempo"].to_numpy(dtype=np.float32)[None, :]
        diff = np.abs(a - b)
        ok = (diff <= tempo_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "tempo")

    # --- duration ---
    if "duration_ms" in cand.columns and "duration_ms" in hist.columns:
        dur_eps = float(rules.get("duration_eps", 12000.0))
        a = cand["duration_ms"].to_numpy(dtype=np.float32)[:, None]
        b = hist["duration_ms"].to_numpy(dtype=np.float32)[None, :]
        diff = np.abs(a - b)
        ok = (diff <= dur_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "duration_ms")

    # --- time_signature exact ---
    if "time_signature" in cand.columns and "time_signature" in hist.columns:
        a = cand["time_signature"].to_numpy(dtype=np.float32)[:, None]
        b = hist["time_signature"].to_numpy(dtype=np.float32)[None, :]
        ok = (a == b) & np.isfinite(a) & np.isfinite(b)
        score += ok.astype(np.float32) * _w(weights, "time_signature")

    # --- explicit exact ---
    if "explicit" in cand.columns and "explicit" in hist.columns:
        a = cand["explicit"].to_numpy(dtype=np.int32)[:, None]
        b = hist["explicit"].to_numpy(dtype=np.int32)[None, :]
        ok = (a == b)
        score += ok.astype(np.float32) * _w(weights, "explicit")

    # --- year ---
    if "year" in cand.columns and "year" in hist.columns:
        year_eps = float(rules.get("year_eps", 3.0))
        a = cand["year"].to_numpy(dtype=np.float32)[:, None]
        b = hist["year"].to_numpy(dtype=np.float32)[None, :]
        diff = np.abs(a - b)
        ok = (diff <= year_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "year")

    # --- artists intersection ---
    min_inter = int(rules.get("artists_min_intersection", 1))
    if min_inter > 0 and ("artists" in cand.columns) and ("artists" in hist.columns):
        cand_sets = [set(parse_listlike(v)) for v in cand["artists"].tolist()]
        hist_sets = [set(parse_listlike(v)) for v in hist["artists"].tolist()]
        wa = _w(weights, "artists")
        # (Nc*Nh) küçük (BASE_SIZE küçük) -> loop burada kabul edilebilir
        for i in range(Nc):
            if not cand_sets[i]:
                continue
            si = cand_sets[i]
            for j in range(Nh):
                if len(si.intersection(hist_sets[j])) >= min_inter:
                    score[i, j] += wa

    # edge weight = round(score) (int)
    w_mat = np.rint(score).astype(np.int32)
    sim_sums = w_mat.sum(axis=1).astype(float)  # her candidate için history toplamı
    return sim_sums


# ---------------------------------------------------------
# 3) TOP 10 TABLO PNG
# ---------------------------------------------------------
def save_top10_table_image_rf(res_df: pd.DataFrame, out_png: str,
                              algo_name: str, dataset_size: int, base_size: int):
    cols = [
        "candidate_name", "candidate_artists",
        "rf_prob_like", "sim_to_first_base_sum",
        "sim_norm_0_1", "final_score"
    ]
    top = res_df[cols].head(10).copy()

    top["rf_prob_like"] = top["rf_prob_like"].round(6)
    top["sim_to_first_base_sum"] = top["sim_to_first_base_sum"].round(3)
    top["sim_norm_0_1"] = top["sim_norm_0_1"].round(6)
    top["final_score"] = top["final_score"].round(6)

    fig_w = 22
    fig_h = 0.75 * (len(top) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}\nTOP 10 ÖNERİ (Random Forest)"
    plt.suptitle(title, fontsize=16, y=0.98)

    table = ax.table(
        cellText=top.values,
        colLabels=top.columns,
        loc="center",
        cellLoc="left",
        colLoc="left"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_linewidth(0.8)

    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# 4) TOP10 GRAF (sadece TOP10 için)
# ---------------------------------------------------------
def build_similarity_matrix_rule_based_window(
    df: pd.DataFrame,
    indices: List[int],
    rules: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Küçük pencere (BASE_SIZE) için full (k×k) üretmek OK.
    """
    sub = df.loc[indices].copy()

    for c in NUMERIC_COLS_RULE:
        if c in sub.columns:
            sub[c] = sub[c].apply(safe_to_float)

    if "explicit" in sub.columns:
        sub["explicit"] = sub["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    if "artists" in sub.columns:
        artists_list = [parse_listlike(v) for v in sub["artists"].tolist()]
    else:
        artists_list = [[] for _ in range(len(indices))]

    k = len(indices)
    score = np.zeros((k, k), dtype=np.float32)

    audio_eps = float(rules.get("audio_eps", 0.10))
    for c in AUDIO_01_COLS:
        if c not in sub.columns:
            continue
        x = sub[c].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= audio_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, c)

    if "key" in sub.columns:
        key_eps = float(rules.get("key_eps", 1.0))
        x = sub["key"].to_numpy(dtype=np.float32)
        d = circular_key_distance(x[:, None], x[None, :])
        ok = (d <= key_eps) & np.isfinite(d)
        score += ok.astype(np.float32) * _w(weights, "key")

    if "mode" in sub.columns:
        x = sub["mode"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * _w(weights, "mode")

    if "loudness" in sub.columns:
        loud_eps = float(rules.get("loudness_eps", 2.0))
        x = sub["loudness"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= loud_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "loudness")

    if "tempo" in sub.columns:
        tempo_eps = float(rules.get("tempo_eps", 6.0))
        x = sub["tempo"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= tempo_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "tempo")

    if "duration_ms" in sub.columns:
        dur_eps = float(rules.get("duration_eps", 12000.0))
        x = sub["duration_ms"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= dur_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "duration_ms")

    if "time_signature" in sub.columns:
        x = sub["time_signature"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * _w(weights, "time_signature")

    if "explicit" in sub.columns:
        x = sub["explicit"].to_numpy(dtype=np.int32)
        ok = (x[:, None] == x[None, :])
        score += ok.astype(np.float32) * _w(weights, "explicit")

    if "year" in sub.columns:
        year_eps = float(rules.get("year_eps", 3.0))
        x = sub["year"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= year_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * _w(weights, "year")

    min_inter = int(rules.get("artists_min_intersection", 1))
    if min_inter > 0:
        wa = _w(weights, "artists")
        for i in range(k):
            ai = set(artists_list[i])
            for j in range(i + 1, k):
                if len(ai.intersection(artists_list[j])) >= min_inter:
                    score[i, j] += wa
                    score[j, i] += wa

    np.fill_diagonal(score, 0.0)
    return np.rint(score).astype(np.int32)


def draw_window_graph_rf(indices: List[int],
                         df: pd.DataFrame,
                         weight_mat: np.ndarray,
                         name_col: str,
                         cand_idx: int,
                         cand_prob: float,
                         out_png: str,
                         algo_name: str,
                         dataset_size: int,
                         base_size: int,
                         top_rank: int,
                         candidate_name: str,
                         edge_label_limit: int = 5000):
    G = nx.Graph()

    for global_idx in indices:
        full_name = str(df.loc[global_idx, name_col]) if name_col in df.columns else str(global_idx)
        label = full_name[:17] + "..." if len(full_name) > 20 else full_name
        is_candidate = (global_idx == cand_idx)
        G.add_node(global_idx, label=label, is_candidate=is_candidate)

    k = len(indices)
    for i in range(k):
        for j in range(i + 1, k):
            w = int(weight_mat[i, j])
            if w > 0:
                G.add_edge(indices[i], indices[j], weight=w)

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.5, iterations=40, seed=42)

    node_wdeg = {n: 0.0 for n in G.nodes}
    for u, v, d in G.edges(data=True):
        node_wdeg[u] += d["weight"]
        node_wdeg[v] += d["weight"]

    node_sizes = [node_wdeg[n] * 30.0 + 250.0 for n in G.nodes]
    node_colors = [1 if G.nodes[n]["is_candidate"] else 0 for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           cmap=plt.cm.coolwarm, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    labels = {}
    for n in G.nodes:
        if G.nodes[n]["is_candidate"]:
            labels[n] = f'{G.nodes[n]["label"]}\n(CAND) RF P(like)={cand_prob:.3f}'
        else:
            labels[n] = G.nodes[n]["label"]
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    if G.number_of_edges() <= edge_label_limit:
        edge_labels = {(u, v): int(d["weight"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.6)

    cand_short = (candidate_name[:35] + "...") if (candidate_name and len(candidate_name) > 38) else candidate_name
    title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}\nTOP {top_rank}: {cand_short}"
    plt.title(title)

    plt.axis("off")
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 5) ANA AKIŞ (OPTİMİZE)
# ---------------------------------------------------------
def main():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "shakira.csv")  # değiştir

    BASE_SIZE = 25
    TOP_K = 10
    SEED = 42

    # Skor ağırlıkları
    ALPHA = 0.80  # RF P(like)
    BETA  = 0.20  # sim_norm (0-1)

    # Grafı sadeleştirmek için (0 kapalı). Örn 4..7 iyi olur.
    EDGE_THRESHOLD = 7

    # RF hız ayarları (büyük veride fark eder)
    RF_TREES = 250           # 400 -> 250 (hız)
    RF_MAX_DEPTH = 18        # None -> 18 (hız)
    RF_MIN_LEAF = 2          # 1 -> 2 (hız)
    RF_MAX_FEATURES = "sqrt" # hız + genelde iyi
    NEG_MULT = 12            # negatif örnek sayısı = BASE_SIZE * NEG_MULT (downsample)

    # progress bar aç/kapat
    PROGRESS = True

    ALGO_NAME = "Random Forest (Single Fit) + Rule-based Similarity (Fast)"
    name_col = "name"
    artist_col = "artists"

    rules = {
        "audio_eps": 0.10,
        "key_eps": 1.0,
        "loudness_eps": 2.0,
        "tempo_eps": 6.0,
        "duration_eps": 12000.0,
        "year_eps": 3.0,
        "artists_min_intersection": 1
    }

    weights = {
        "danceability": 1.0,
        "energy": 1.0,
        "speechiness": 1.0,
        "acousticness": 1.0,
        "instrumentalness": 1.0,
        "liveness": 1.0,
        "valence": 1.0,
        "key": 1.0,
        "mode": 1.0,
        "loudness": 1.0,
        "tempo": 1.0,
        "duration_ms": 1.0,
        "time_signature": 1.0,
        "explicit": 1.0,
        "year": 1.0,
        "artists": 1.0,
    }

    t0 = time.time()
    df = load_data_forced_header(file_path)

    possible_features = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    feature_cols = [c for c in possible_features if c in df.columns]
    if len(feature_cols) == 0:
        print("KRİTİK HATA: Feature sütunu bulunamadı.")
        sys.exit(1)

    # Numeric parse
    print("Veriler sayısal formata çevriliyor...")
    needed_numeric = set(feature_cols + ["duration_ms", "time_signature", "year"] + AUDIO_01_COLS)
    for col in needed_numeric:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    n = len(df)
    print(f"Toplam kayıt: {n}")
    print(f"Özellik sayısı: {len(feature_cols)}")
    print(f"BASE_SIZE: {BASE_SIZE}")

    if n < BASE_SIZE + 1:
        print(f"KRİTİK HATA: BASE_SIZE={BASE_SIZE} ama veri sayısı {n}. En az BASE_SIZE+1 kayıt olmalı.")
        sys.exit(1)

    history = list(range(BASE_SIZE))
    candidates = list(range(BASE_SIZE, n))
    fixed_base = list(range(BASE_SIZE - 1))

    # --- A) X standardize ---
    X = df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # --- B) RF'yi 1 kez eğit (downsample negative) ---
    print("\n--- ADIM B: RandomForest tek sefer eğitiliyor (hız için negative downsample) ---")
    rng = np.random.default_rng(SEED)

    pos_idx = np.array(history, dtype=int)
    neg_pool = np.array(candidates, dtype=int)

    neg_count = min(len(neg_pool), int(len(pos_idx) * NEG_MULT))
    neg_idx = rng.choice(neg_pool, size=neg_count, replace=False)

    train_idx = np.concatenate([pos_idx, neg_idx])
    y_train = np.zeros(len(train_idx), dtype=int)
    y_train[:len(pos_idx)] = 1

    clf = RandomForestClassifier(
        n_estimators=RF_TREES,
        random_state=SEED,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF,
        min_samples_split=2,
        max_features=RF_MAX_FEATURES,
        class_weight="balanced",
        n_jobs=-1,
        bootstrap=True
    )
    clf.fit(Xs[train_idx], y_train)

    # tüm adaylara batch predict_proba
    proba = clf.predict_proba(Xs[candidates])
    # class 1 sütununu bul
    if 1 in clf.classes_:
        one_col = int(np.where(clf.classes_ == 1)[0][0])
        rf_probs = proba[:, one_col].astype(float)
    else:
        rf_probs = np.zeros(len(candidates), dtype=float)

    # --- C) Rule-based similarity: sadece candidate x history ---
    print("\n--- ADIM C: Benzerlik (candidate × history) hızlı hesaplanıyor ---")
    sim_sums = compute_similarity_sums_candidate_history(
        df=df,
        candidates=candidates,
        history=history,
        rules=rules,
        weights=weights
    )

    # normalize (0-1)
    sim_min = float(np.min(sim_sums)) if len(sim_sums) else 0.0
    sim_max = float(np.max(sim_sums)) if len(sim_sums) else 1.0
    denom = (sim_max - sim_min) if (sim_max - sim_min) != 0 else 1.0
    sim_norm = (sim_sums - sim_min) / denom

    final_scores = ALPHA * rf_probs + BETA * sim_norm

    # --- D) Sonuç dataframe ---
    print("\n--- ADIM D: TOP skorlar çıkarılıyor ---")
    cand_names = (df.loc[candidates, name_col].astype(str).values
                  if name_col in df.columns else np.array([str(i) for i in candidates]))
    cand_artists = (df.loc[candidates, artist_col].astype(str).values
                    if artist_col in df.columns else np.array([""] * len(candidates)))

    res_df = pd.DataFrame({
        "base_size": BASE_SIZE,
        "candidate_global_index": candidates,
        "candidate_name": cand_names,
        "candidate_artists": cand_artists,
        "rf_prob_like": rf_probs,
        "sim_to_first_base_sum": sim_sums,
        "sim_norm_0_1": sim_norm,
        "final_score": final_scores
    }).sort_values("final_score", ascending=False).reset_index(drop=True)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = f"{base_filename}_rf_fast_base{BASE_SIZE}_top{TOP_K}_recommendations.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Sonuç CSV kaydedildi: {out_csv}")

    top_k = min(TOP_K, len(res_df))
    if top_k == 0:
        print("UYARI: Aday bulunamadı.")
        return

    out_dir = os.path.join(base_dir, f"{base_filename}_rf_fast_base{BASE_SIZE}_TOP{top_k}_{n}_size")
    os.makedirs(out_dir, exist_ok=True)

    # --- Sadece TOP10 tablo + TOP10 graf üret ---
    table_png = os.path.join(out_dir, f"TOP_{top_k:02d}_TABLE.png")
    save_top10_table_image_rf(
        res_df=res_df,
        out_png=table_png,
        algo_name=ALGO_NAME,
        dataset_size=n,
        base_size=BASE_SIZE
    )
    print(f"✅ TOP {top_k} tablo PNG kaydedildi: {table_png}")

    print(f"\n--- ADIM E: SADECE TOP {top_k} için graf PNG üretimi ---")
    pb = ProgressBar(top_k, prefix="Graf PNG:", enabled=PROGRESS)

    for rank in range(top_k):
        cand_idx = int(res_df.loc[rank, "candidate_global_index"])
        cand_name = str(res_df.loc[rank, "candidate_name"])
        cand_prob = float(res_df.loc[rank, "rf_prob_like"])

        window_indices = fixed_base + [cand_idx]
        w_window = build_similarity_matrix_rule_based_window(df, window_indices, rules=rules, weights=weights)

        if EDGE_THRESHOLD > 0:
            w_window = (w_window * (w_window >= EDGE_THRESHOLD).astype(np.int32))

        out_png = os.path.join(out_dir, f"TOP_{rank+1:02d}_cand_{cand_idx}.png")
        draw_window_graph_rf(
            indices=window_indices,
            df=df,
            weight_mat=w_window,
            name_col=name_col,
            cand_idx=cand_idx,
            cand_prob=cand_prob,
            out_png=out_png,
            algo_name=ALGO_NAME,
            dataset_size=n,
            base_size=BASE_SIZE,
            top_rank=rank+1,
            candidate_name=cand_name,
            edge_label_limit=5000
        )
        pb.update()

    print(f"\n✅ Sadece TOP {top_k} graf + tablo üretildi: {out_dir}")

    print("\n--- TOP 10 ÖNERİ ---")
    show_cols = [
        "candidate_name", "candidate_artists",
        "rf_prob_like", "sim_to_first_base_sum", "final_score"
    ]
    print(res_df[show_cols].head(10).to_string(index=False))

    print(f"\nToplam süre: {time.time() - t0:.2f} sn")
    print("Not: Hız için RF LOO kaldırıldı (tek fit + downsample).")
    print("Graf kalabalıksa EDGE_THRESHOLD=4..7 deneyebilirsin.")


if __name__ == "__main__":
    main()
