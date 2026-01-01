import os
import sys
import time
import ast
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

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
# 1) VERİ OKUMA (CSV/EXCEL)
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

    # tek kolona yığılmışsa ayır
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

    # header kaymışsa
    if 'danceability' in df.columns and str(df.iloc[0]['danceability']).strip().lower() == 'danceability':
        df = df.iloc[1:]

    df.columns = df.columns.str.strip().str.lower()
    return df.reset_index(drop=True)


# ---------------------------------------------------------
# 2) HELPERS
# ---------------------------------------------------------
AUDIO_01_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence"
]

NUMERIC_COLS = AUDIO_01_COLS + [
    "key", "loudness", "mode", "tempo", "duration_ms",
    "time_signature", "year"
]

BOOL_COLS = ["explicit"]

LISTLIKE_COLS = ["artists"]  # string içinde ['a','b'] gibi

def safe_to_float(s):
    try:
        return float(s)
    except:
        return np.nan

def parse_listlike(x) -> List[str]:
    """
    "['A', 'B']" gibi stringleri listeye çevirir.
    Bozuk formatlarda boş liste döndürür.
    """
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
        # fallback: virgülle ayrılmış sandı
        if "," in sx:
            return [p.strip().strip("'").strip('"') for p in sx.split(",") if p.strip()]
        return [sx.strip().strip("'").strip('"')]

def circular_key_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    key 0-11 dairesel uzaklık.
    """
    d = np.abs(a - b)
    return np.minimum(d, 12 - d)


# ---------------------------------------------------------
# 3) KOLON-BAZLI BENZERLİK MATRİSİ
#    weight = "kaç kriter tuttu" (istersen ağırlık da ekledim)
# ---------------------------------------------------------
def build_similarity_matrix_rule_based(
    df: pd.DataFrame,
    indices: List[int],
    rules: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Kurallar:
      - AUDIO_01: abs diff <= rules["audio_eps"]
      - key: circular dist <= rules["key_eps"]
      - loudness: abs diff <= rules["loudness_eps"]
      - tempo: abs diff <= rules["tempo_eps"]
      - duration_ms: abs diff <= rules["duration_eps"]
      - year: abs diff <= rules["year_eps"]
      - mode/time_signature/explicit: exact match
      - artists: intersection >= rules["artists_min_intersection"]
    """
    sub = df.loc[indices].copy()

    # sayısalları çevir
    for c in NUMERIC_COLS:
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

    # ---- AUDIO 0-1 ----
    audio_eps = float(rules.get("audio_eps", 0.10))
    for c in AUDIO_01_COLS:
        if c not in sub.columns:
            continue
        x = sub[c].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= audio_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w(c)

    # ---- KEY (circular) ----
    if "key" in sub.columns:
        key_eps = float(rules.get("key_eps", 1.0))
        x = sub["key"].to_numpy(dtype=np.float32)
        d = circular_key_distance(x[:, None], x[None, :])
        ok = (d <= key_eps) & np.isfinite(d)
        score += ok.astype(np.float32) * w("key")

    # ---- MODE exact ----
    if "mode" in sub.columns:
        x = sub["mode"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * w("mode")

    # ---- LOUDNESS ----
    if "loudness" in sub.columns:
        loud_eps = float(rules.get("loudness_eps", 2.0))
        x = sub["loudness"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= loud_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("loudness")

    # ---- TEMPO ----
    if "tempo" in sub.columns:
        tempo_eps = float(rules.get("tempo_eps", 6.0))
        x = sub["tempo"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= tempo_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("tempo")

    # ---- DURATION ----
    if "duration_ms" in sub.columns:
        dur_eps = float(rules.get("duration_eps", 12000.0))
        x = sub["duration_ms"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= dur_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("duration_ms")

    # ---- TIME SIGNATURE exact ----
    if "time_signature" in sub.columns:
        x = sub["time_signature"].to_numpy(dtype=np.float32)
        ok = (x[:, None] == x[None, :]) & np.isfinite(x[:, None]) & np.isfinite(x[None, :])
        score += ok.astype(np.float32) * w("time_signature")

    # ---- EXPLICIT exact ----
    if "explicit" in sub.columns:
        x = sub["explicit"].to_numpy(dtype=np.int32)
        ok = (x[:, None] == x[None, :])
        score += ok.astype(np.float32) * w("explicit")

    # ---- YEAR ----
    if "year" in sub.columns:
        year_eps = float(rules.get("year_eps", 3.0))
        x = sub["year"].to_numpy(dtype=np.float32)
        diff = np.abs(x[:, None] - x[None, :])
        ok = (diff <= year_eps) & np.isfinite(diff)
        score += ok.astype(np.float32) * w("year")

    # ---- ARTISTS intersection ----
    min_inter = int(rules.get("artists_min_intersection", 1))
    if min_inter > 0:
        # O(k^2) ama k=10 gibi pencerelerde çok hızlı
        for i in range(k):
            ai = set(artists_list[i])
            for j in range(i + 1, k):
                inter = len(ai.intersection(artists_list[j]))
                if inter >= min_inter:
                    score[i, j] += w("artists")
                    score[j, i] += w("artists")

    np.fill_diagonal(score, 0.0)

    # integer edge weight istiyorsun: yakın kriter sayısı gibi
    return np.rint(score).astype(np.int32)


# ---------------------------------------------------------
# 4) AĞIRLIKLI MALATYA CENTRALITY
# ---------------------------------------------------------
def malatya_centrality_weighted(weight_mat: np.ndarray, threshold: int):
    mask = (weight_mat >= threshold).astype(np.int32)
    W = (weight_mat * mask).astype(np.float64)

    wdeg = W.sum(axis=1)
    n = W.shape[0]
    cent = np.zeros(n, dtype=np.float64)

    active = np.where(wdeg > 0)[0]
    pb = ProgressBar(len(active), prefix="Düğümler:  ")
    for i in active:
        neighbors = np.where(W[i] > 0)[0]
        if neighbors.size == 0:
            pb.update()
            continue
        denom = wdeg[neighbors]
        valid = denom > 0
        if np.any(valid):
            cent[i] = np.sum(wdeg[i] / denom[valid])
        pb.update()

    return cent, wdeg


# ---------------------------------------------------------
# 5) GRAF ÇİZ
# ---------------------------------------------------------
def draw_window_graph(indices: List[int], df: pd.DataFrame, weight_mat: np.ndarray,
                      cent: np.ndarray, wdeg: np.ndarray, name_col: str, out_png: str,
                      algo_name: str, dataset_size: int, base_size: int,
                      top_rank: Optional[int] = None,
                      candidate_name: Optional[str] = None,
                      edge_label_limit: int = 3000):
    G = nx.Graph()

    for local_i, global_idx in enumerate(indices):
        full_name = str(df.loc[global_idx, name_col]) if name_col in df.columns else str(global_idx)
        label = full_name[:17] + "..." if len(full_name) > 20 else full_name
        G.add_node(global_idx, label=label, cent=float(cent[local_i]), wdeg=float(wdeg[local_i]))

    k = len(indices)
    for i in range(k):
        for j in range(i + 1, k):
            w = int(weight_mat[i, j])
            if w > 0:
                G.add_edge(indices[i], indices[j], weight=w)

    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    node_sizes = [(G.nodes[n]["cent"] * 20.0 + 200.0) for n in G.nodes]
    node_colors = [G.nodes[n]["wdeg"] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                           cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    labels = {n: f'{G.nodes[n]["label"]}\n(wdeg:{int(G.nodes[n]["wdeg"])})' for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    if G.number_of_edges() <= edge_label_limit:
        edge_labels = {(u, v): int(d["weight"]) for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, alpha=0.6)

    base_title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}"
    if top_rank is not None:
        cand_short = (candidate_name[:35] + "...") if (candidate_name and len(candidate_name) > 38) else (candidate_name or "")
        title = f"{base_title}\nTOP {top_rank}: {cand_short}"
    else:
        title = base_title

    plt.title(title)
    plt.axis("off")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# 6) TOP 10 TABLO PNG
# ---------------------------------------------------------
def save_top10_table_image(res_df: pd.DataFrame, out_png: str,
                           algo_name: str, dataset_size: int, base_size: int):
    cols = [
        "candidate_name", "candidate_artists",
        "malatya_centrality_weighted", "sim_to_first_base_sum", "final_score"
    ]
    top = res_df[cols].head(10).copy()

    top["malatya_centrality_weighted"] = top["malatya_centrality_weighted"].round(6)
    top["sim_to_first_base_sum"] = top["sim_to_first_base_sum"].round(3)
    top["final_score"] = top["final_score"].round(6)

    fig_w = 22
    fig_h = 0.75 * (len(top) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    title = f"{algo_name} | Dataset n={dataset_size} | Graph BASE_SIZE={base_size}\nTOP 10 ÖNERİ (Adaylar içinden)"
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
# 7) ANA AKIŞ
# ---------------------------------------------------------
def main():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "shakira.csv")

    BASE_SIZE = 25
    TOP_K = 10


    THRESHOLD = 6

    ALPHA = 0.60
    BETA = 0.40
    ALGO_NAME = "Malatya Centrality (Rule-based Similarity)"

    name_col = "name"
    artist_col = "artists"

    # --- Benzerlik kuralları (kolon bazlı) ---
    rules = {
        "audio_eps": 0.10,            # 0-1 arası feature farkı 
        "key_eps": 1.0,               # circular key dist
        "loudness_eps": 2.0,          # dB
        "tempo_eps": 6.0,             # BPM
        "duration_eps": 12000.0,      # ms
        "year_eps": 3.0,              # yıl farkı
       
    }

    # İstersen bazı kriterleri daha önemli yap:
    weights = {
        # audio 0-1
        "danceability": 1.0,
        "energy": 1.0,  
        "speechiness": 0.8,
        "acousticness": 0.8,
        "instrumentalness": 0.8,
        "liveness": 0.8,
        "valence": 1.0,

        # diğerleri
        "key": 1.0,
        "mode": 0.8,
        "loudness": 1.0,
        "tempo": 1.0,
        "duration_ms": 0.6,
        "time_signature": 0.4,
        "explicit": 0.4,
        "year": 0.5,
        "artists": 1.2,
    }

    df = load_data_forced_header(file_path)

    # sayısalları normalize etmeden numeric'e çeviriyoruz (kural bazlı)
    print("Veriler sayısal formata çevriliyor...")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace("'", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # explicit
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    # minimum gerekli kolonlar yoksa bile devam edebilir; sadece olanlarla skorlar.
    df = df.reset_index(drop=True)
    n = len(df)

    print(f"Toplam kayıt: {n}")
    print(f"BASE_SIZE (graf düğüm sayısı): {BASE_SIZE}")

    if n < BASE_SIZE + 1:
        print(f"KRİTİK HATA: BASE_SIZE={BASE_SIZE} seçtin ama veri sayısı {n}. En az BASE_SIZE+1 kayıt olmalı.")
        sys.exit(1)

    history_indices = list(range(BASE_SIZE))
    fixed_indices = list(range(BASE_SIZE - 1))
    candidates = list(range(BASE_SIZE, n))

    print("\n--- ADIM A: Global benzerlik matrisi (n x n) hesaplanıyor ---")
    # Büyük datasetlerde bu n^2 pahalı olabilir.
    # Şimdilik senin akışınla uyumlu tutuyorum.
    global_w = build_similarity_matrix_rule_based(df, list(range(n)), rules=rules, weights=weights)

    results = []
    print(f"\n--- ADIM B: ({BASE_SIZE-1}) sabit + 1 slot ile aday denemeleri ---")
    pb = ProgressBar(len(candidates), prefix="Adaylar:   ")

    for cand_idx in candidates:
        window_indices = fixed_indices + [cand_idx]

        w_window = build_similarity_matrix_rule_based(df, window_indices, rules=rules, weights=weights)

        cent, wdeg = malatya_centrality_weighted(w_window, threshold=THRESHOLD)

        cand_local = len(window_indices) - 1
        cand_cent = float(cent[cand_local])
        cand_wdeg = float(wdeg[cand_local])

        sim_to_history = float(global_w[cand_idx, history_indices].sum())
        score = ALPHA * cand_cent + BETA * sim_to_history

        results.append({
            "base_size": BASE_SIZE,
            "candidate_global_index": cand_idx,
            "candidate_name": str(df.loc[cand_idx, name_col]) if name_col in df.columns else str(cand_idx),
            "candidate_artists": str(df.loc[cand_idx, artist_col]) if artist_col in df.columns else "",
            "malatya_centrality_weighted": cand_cent,
            "weighted_degree": cand_wdeg,
            "sim_to_first_base_sum": sim_to_history,
            "final_score": score
        })
        pb.update()

    res_df = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)

    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = f"{base_filename}_base{BASE_SIZE}_rolling_recommendations.csv"
    res_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Öneri sonuçları kaydedildi: {out_csv}")

    top_k = min(TOP_K, len(res_df))
    out_dir = os.path.join(base_dir, f"{base_filename}_rulebased_base{BASE_SIZE}_top{top_k}_graphs_{n}_size")
    os.makedirs(out_dir, exist_ok=True)

    # TOP10 tablo
    table_png = os.path.join(out_dir, f"TOP_{top_k:02d}_TABLE.png")
    save_top10_table_image(
        res_df=res_df,
        out_png=table_png,
        algo_name=ALGO_NAME,
        dataset_size=n,
        base_size=BASE_SIZE
    )
    print(f"✅ TOP {top_k} tablo PNG kaydedildi: {table_png}")

    # TOP K graf
    print(f"\n--- ADIM C: TOP {top_k} için PNG graf üretimi ---")
    pb2 = ProgressBar(top_k, prefix="Graf PNG:  ")

    for rank in range(top_k):
        cand_idx = int(res_df.loc[rank, "candidate_global_index"])
        cand_name = str(res_df.loc[rank, "candidate_name"])

        window_indices = fixed_indices + [cand_idx]
        w_window = build_similarity_matrix_rule_based(df, window_indices, rules=rules, weights=weights)

        cent, wdeg = malatya_centrality_weighted(w_window, threshold=THRESHOLD)
        w_for_draw = (w_window * (w_window >= THRESHOLD).astype(int))

        out_png = os.path.join(out_dir, f"TOP_{rank+1:02d}_cand_{cand_idx}.png")
        draw_window_graph(
            indices=window_indices,
            df=df,
            weight_mat=w_for_draw,
            cent=cent,
            wdeg=wdeg,
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

    print(f"\n✅ TOP {top_k} graf PNG'leri + tablo PNG klasörde: {out_dir}")

    print("\n--- TOP 10 ÖNERİ (Adaylar içinden) ---")
    show_cols = [
        "candidate_name", "candidate_artists",
        "malatya_centrality_weighted", "sim_to_first_base_sum", "final_score"
    ]
    print(res_df[show_cols].head(10).to_string(index=False))

    print("\nNot: Benzerlik edge-weight'i artık 'tutan kriter sayısı/ağırlığı'.")
    print(f"THRESHOLD={THRESHOLD} bu yüzden 4-7 bandında güzel çalışır.")


if __name__ == "__main__":
    main()
