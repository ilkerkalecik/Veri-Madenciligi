import os
import sys
import time
import ast
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# İLERLEME ÇUBUĞU SINIFI
# ---------------------------------------------------------
class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=30, fill='█'):
        self.total = max(int(total), 1)
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0
        self.start_time = time.time()

    def update(self, progress=1):
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
# 2) RULE-BASED SIMILARITY (senin kuralların)
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

def build_similarity_matrix_rule_based(
    df: pd.DataFrame,
    indices: List[int],
    rules: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    sub = df.loc[indices].copy()

    # numeric parse
    for c in NUMERIC_COLS_RULE:
        if c in sub.columns:
            sub[c] = sub[c].apply(safe_to_float)

    # explicit -> 0/1
    if "explicit" in sub.columns:
        sub["explicit"] = sub["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    # artists parse
    if "artists" in sub.columns:
        artists_list = [parse_listlike(v) for v in sub["artists"].tolist()]
    else:
        artists_list = [[] for _ in range(len(indices))]

    k = len(indices)
    score = np.zeros((k, k), dtype=np.float32)

    def w(name: str) -> float:
        return float(weights.get(name, 1.0)) if weights else 1.0

    # --- 0-1 audio: abs diff <= 0.10 ---
    audio_eps = float(rules.get("audio_eps", 0.10))
    for c in AUDIO_01_COLS:
        if c not in sub.columns:
            continue
        x = sub[c].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= audio_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w(c)

    # --- key circular <= 1 ---
    if "key" in sub.columns:
        key_eps = float(rules.get("key_eps", 1.0))
        x = sub["key"].to_numpy(dtype=np.float32)
        d = circular_key_distance(x[:, None], x[None, :])
        ok = (d <= key_eps) & np.isfinite(d)
        score += ok.astype(np.float32) * w("key")

    # --- mode exact ---
    if "mode" in sub.columns:
        x = sub["mode"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * w("mode")

    # --- loudness <= 2 dB ---
    if "loudness" in sub.columns:
        loud_eps = float(rules.get("loudness_eps", 2.0))
        x = sub["loudness"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= loud_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("loudness")

    # --- tempo <= 6 ---
    if "tempo" in sub.columns:
        tempo_eps = float(rules.get("tempo_eps", 6.0))
        x = sub["tempo"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= tempo_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("tempo")

    # --- duration <= 12000 ms ---
    if "duration_ms" in sub.columns:
        dur_eps = float(rules.get("duration_eps", 12000.0))
        x = sub["duration_ms"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= dur_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("duration_ms")

    # --- time_signature exact ---
    if "time_signature" in sub.columns:
        x = sub["time_signature"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * w("time_signature")

    # --- explicit exact ---
    if "explicit" in sub.columns:
        x = sub["explicit"].to_numpy(dtype=np.int32)
        ok = (x[:, None] == x[None, :])
        score += ok.astype(np.float32) * w("explicit")

    # --- year <= 3 ---
    if "year" in sub.columns:
        year_eps = float(rules.get("year_eps", 3.0))
        x = sub["year"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= year_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("year")

    # --- artists intersection > 0 ---
    min_inter = int(rules.get("artists_min_intersection", 1))
    if min_inter > 0:
        for i in range(k):
            ai = set(artists_list[i])
            for j in range(i + 1, k):
                inter = len(ai.intersection(artists_list[j]))
                if inter >= min_inter:
                    score[i, j] += w("artists")
                    score[j, i] += w("artists")

    np.fill_diagonal(score, 0.0)
    return np.rint(score).astype(np.int32)

# ---------------------------------------------------------
# 3) GRAF ÇİZ (KMeans)
# ---------------------------------------------------------
def draw_window_graph_kmeans(indices: List[int],
                             df: pd.DataFrame,
                             weight_mat: np.ndarray,
                             clusters_global: np.ndarray,
                             name_col: str,
                             out_png: str,
                             algo_name: str,
                             dataset_size: int,
                             base_size: int,
                             top_rank: int,
                             candidate_name: str,
                             edge_label_limit: int = 3000):
    G = nx.Graph()

    for global_idx in indices:
        full_name = str(df.loc[global_idx, name_col]) if name_col in df.columns else str(global_idx)
        label = full_name[:17] + "..." if len(full_name) > 20 else full_name
        cl = int(clusters_global[global_idx])
        G.add_node(global_idx, label=label, cluster=cl)

    k = len(indices)
    for i in range(k):
        for j in range(i + 1, k):
            w = int(weight_mat[i, j])
            if w > 0:
                G.add_edge(indices[i], indices[j], weight=w)

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    node_wdeg = {n: 0.0 for n in G.nodes}
    for u, v, d in G.edges(data=True):
        node_wdeg[u] += d["weight"]
        node_wdeg[v] += d["weight"]

    node_sizes = [node_wdeg[n] * 30.0 + 250.0 for n in G.nodes]
    node_colors = [G.nodes[n]["cluster"] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.tab10, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    labels = {n: f'{G.nodes[n]["label"]}\n(C:{G.nodes[n]["cluster"]})' for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    if G.number_of_edges() <= edge_label_limit:
        edge_labels = {(u, v): int(d["weight"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.6)

    cand_short = (candidate_name[:35] + "...") if (candidate_name and len(candidate_name) > 38) else candidate_name
    title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}\nTOP {top_rank}: {cand_short}"

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------
# 4) TOP 10 TABLO PNG (KMeans)
# ---------------------------------------------------------
def save_top10_table_image_kmeans(res_df: pd.DataFrame, out_png: str,
                                  algo_name: str, dataset_size: int, base_size: int):
    cols = [
        "candidate_name", "candidate_artists",
        "kmeans_cluster", "cluster_affinity",
        "sim_to_first_base_sum", "final_score"
    ]
    top = res_df[cols].head(10).copy()

    top["cluster_affinity"] = top["cluster_affinity"].round(6)
    top["sim_to_first_base_sum"] = top["sim_to_first_base_sum"].round(3)
    top["final_score"] = top["final_score"].round(6)

    fig_w = 22
    fig_h = 0.75 * (len(top) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}\nTOP 10 ÖNERİ (KMeans)"
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

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------
# 5) ANA AKIŞ
# ---------------------------------------------------------
def main():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "shakira.csv")

    BASE_SIZE = 25
    TOP_K = 10

    N_CLUSTERS = 6
    SEED = 42

    ALPHA = 0.60  # cluster_affinity
    BETA = 0.40   # sim_to_history

    # Graf çok kalabalıksa edge filtresi:
    THRESHOLD_FOR_EDGES = 7  # 0=filtre yok, örn 4-6 yapabilirsin

    ALGO_NAME = "KMeans (Global) + Rolling Slot [Rule-based Similarity]"
    name_col = "name"
    artist_col = "artists"

    # --- RULES (SENİN KURALLAR) ---
    rules = {
        "audio_eps": 0.10,
        "key_eps": 1.0,
        "loudness_eps": 2.0,
        "tempo_eps": 6.0,
        "duration_eps": 12000.0,
        "year_eps": 3.0,
        "artists_min_intersection": 1
    }

    # --- WEIGHTS (opsiyonel, istersen 1.0 yapıp sadeleştirebilirsin) ---
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

    df = load_data_forced_header(file_path)

    # KMeans feature seti (cluster için)
    possible_features = [
        "danceability", "energy", "key", "loudness", "mode",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]
    feature_cols = [c for c in possible_features if c in df.columns]
    if len(feature_cols) == 0:
        print("KRİTİK HATA: Feature sütunu bulunamadı.")
        sys.exit(1)

    # numeric parse (hem KMeans hem similarity için)
    print("Veriler sayısal formata çevriliyor...")
    for col in set(feature_cols + ["duration_ms", "time_signature", "year", "key", "loudness", "tempo", "mode"] + AUDIO_01_COLS):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # explicit normalize
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    # KMeans için NaN temizliği (cluster eğitimi bozulmasın)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    n = len(df)
    print(f"Toplam kayıt: {n}")
    print(f"Özellik sayısı (KMeans): {len(feature_cols)}")
    print(f"BASE_SIZE (graf düğüm sayısı): {BASE_SIZE}")

    if n < BASE_SIZE + 1:
        print(f"KRİTİK HATA: BASE_SIZE={BASE_SIZE} seçtin ama veri sayısı {n}. En az BASE_SIZE+1 kayıt olmalı.")
        sys.exit(1)

    print("\n--- ADIM A: Global benzerlik matrisi (n x n) [RULE-BASED] ---")
    global_w = build_similarity_matrix_rule_based(df, list(range(n)), rules=rules, weights=weights)

    print("\n--- ADIM B: KMeans eğitiliyor (GLOBAL) ---")
    X = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kk = min(N_CLUSTERS, max(2, n // 2))
    if kk != N_CLUSTERS:
        print(f"UYARI: n={n} olduğu için N_CLUSTERS={N_CLUSTERS} yerine k={kk} kullanıldı.")

    kmeans = KMeans(n_clusters=kk, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(Xs)

    user_history = list(range(BASE_SIZE))
    user_hist_clusters = clusters[user_history]
    user_cluster_hist = np.bincount(user_hist_clusters, minlength=kk)

    fixed_base = list(range(BASE_SIZE - 1))
    candidates = list(range(BASE_SIZE, n))

    results = []
    print(f"\n--- ADIM C: ({BASE_SIZE-1}) sabit + 1 slot ile KMeans öneri ---")
    pb = ProgressBar(len(candidates), prefix="Adaylar:   ")

    for cand_idx in candidates:
        cand_cluster = int(clusters[cand_idx])
        cluster_affinity = float(user_cluster_hist[cand_cluster]) / float(len(user_history))

        sim_to_history = float(global_w[cand_idx, user_history].sum())
        score = ALPHA * cluster_affinity + BETA * sim_to_history

        results.append({
            "base_size": BASE_SIZE,
            "candidate_global_index": cand_idx,
            "candidate_name": str(df.loc[cand_idx, name_col]) if name_col in df.columns else str(cand_idx),
            "candidate_artists": str(df.loc[cand_idx, artist_col]) if artist_col in df.columns else "",
            "kmeans_cluster": cand_cluster,
            "cluster_affinity": cluster_affinity,
            "sim_to_first_base_sum": sim_to_history,
            "final_score": score
        })
        pb.update()

    res_df = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = f"{base_filename}_kmeans_rulebased_base{BASE_SIZE}_rolling_recommendations.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ KMeans öneri sonuçları kaydedildi: {out_csv}")

    top_k = min(TOP_K, len(res_df))
    if top_k == 0:
        print("UYARI: Aday bulunamadı.")
        return

    out_dir = os.path.join(base_dir, f"{base_filename}_kmeans_rulebased_base{BASE_SIZE}_top{top_k}_graphs_{n}_size")
    os.makedirs(out_dir, exist_ok=True)

    # 1) TOP 10 tablo PNG
    table_png = os.path.join(out_dir, f"TOP_{top_k:02d}_TABLE.png")
    save_top10_table_image_kmeans(
        res_df=res_df,
        out_png=table_png,
        algo_name=ALGO_NAME,
        dataset_size=n,
        base_size=BASE_SIZE
    )
    print(f"✅ TOP {top_k} tablo PNG kaydedildi: {table_png}")

    # 2) TOP 10 graf PNG
    print(f"\n--- ADIM D: TOP {top_k} için graf PNG üretimi ---")
    pb2 = ProgressBar(top_k, prefix="Graf PNG:  ")

    for rank in range(top_k):
        cand_idx = int(res_df.loc[rank, "candidate_global_index"])
        cand_name = str(res_df.loc[rank, "candidate_name"])

        window_indices = fixed_base + [cand_idx]
        w_window = build_similarity_matrix_rule_based(df, window_indices, rules=rules, weights=weights)

        if THRESHOLD_FOR_EDGES > 0:
            w_window = (w_window * (w_window >= THRESHOLD_FOR_EDGES).astype(np.int32))

        out_png = os.path.join(out_dir, f"TOP_{rank+1:02d}_cand_{cand_idx}.png")
        draw_window_graph_kmeans(
            indices=window_indices,
            df=df,
            weight_mat=w_window,
            clusters_global=clusters,
            name_col=name_col,
            out_png=out_png,
            algo_name=ALGO_NAME,
            dataset_size=n,
            base_size=BASE_SIZE,
            top_rank=rank+1,
            candidate_name=cand_name,
            edge_label_limit=5000
        )
        pb2.update()

    print(f"\n✅ TOP {top_k} graf PNG + tablo PNG klasöre kaydedildi: {out_dir}")

    print("\n--- TOP 10 ÖNERİ (KMeans / Rule-based Similarity) ---")
    show_cols = [
        "candidate_name", "candidate_artists",
        "kmeans_cluster", "cluster_affinity",
        "sim_to_first_base_sum", "final_score"
    ]
    print(res_df[show_cols].head(10).to_string(index=False))

    print("\nNot: similarity edge-weight'i artık 'tutan kuralların ağırlık toplamı' (int).")
    print("Graf kalabalıksa THRESHOLD_FOR_EDGES=4..6 deneyebilirsin.")


if __name__ == "__main__":
    main()
