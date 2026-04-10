import argparse
import os
import sys

import numpy as np

from recommender_utils import (
    artist_overlap_score,
    build_numeric_feature_matrix,
    build_result_frame,
    listened_artist_set,
    load_dataset,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:
    print("HATA: PyTorch import edilemedi.")
    print("Kurulum: python3 -m pip install torch")
    print(f"Detay: {exc}")
    sys.exit(1)


class SequenceContextDataset(Dataset):
    def __init__(self, features: np.ndarray, context_size: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.context_size = context_size

    def __len__(self) -> int:
        return len(self.features) - self.context_size

    def __getitem__(self, index: int):
        start = index
        end = index + self.context_size
        context = self.features[start:end]
        target = self.features[end]
        return context, target


class SongEmbeddingModel(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return F.normalize(embeddings, dim=-1)

    def forward(self, context: torch.Tensor, target: torch.Tensor):
        context_embeddings = self.encode(context)
        target_embeddings = self.encode(target)
        context_profile = F.normalize(context_embeddings.mean(dim=1), dim=-1)
        logits = torch.matmul(context_profile, target_embeddings.T) / 0.1
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss


def train_embedding_model(features: np.ndarray, embedding_dim: int, context_size: int, epochs: int, batch_size: int):
    dataset = SequenceContextDataset(features, context_size=context_size)
    if len(dataset) < 2:
        raise ValueError("Embedding modeli icin yeterli ardisk sarki yok.")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    model = SongEmbeddingModel(input_dim=features.shape[1], embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for context, target in loader:
            optimizer.zero_grad()
            loss = model(context, target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        all_features = torch.tensor(features, dtype=torch.float32)
        embeddings = model.encode(all_features).cpu().numpy()

    return embeddings


def cosine_scores(candidates: np.ndarray, profile: np.ndarray) -> np.ndarray:
    numerator = np.sum(candidates * profile, axis=1)
    denom = np.linalg.norm(candidates, axis=1) * np.linalg.norm(profile)
    denom = np.where(denom == 0.0, 1e-8, denom)
    return numerator / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch embedding tabanli sarki onerici")
    parser.add_argument("--file", default="shakira.csv", help="CSV veya Excel veri dosyasi")
    parser.add_argument("--base-size", type=int, default=25, help="Ilk kac sarki dinlenmis kabul edilsin")
    parser.add_argument("--top-k", type=int, default=10, help="Kac sarki onerilsin")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding boyutu")
    parser.add_argument("--context-size", type=int, default=5, help="Bir sonraki sarki tahmini icin kullanilan onceki sarki sayisi")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch sayisi")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch boyutu")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    file_path = args.file if os.path.isabs(args.file) else os.path.join(base_dir, args.file)

    try:
        df = load_dataset(file_path)
        if len(df) <= args.base_size:
            raise ValueError(f"Veri sayisi {len(df)}. En az {args.base_size + 1} sarki olmali.")

        features, _ = build_numeric_feature_matrix(df)
        embeddings = train_embedding_model(
            features=features,
            embedding_dim=args.embedding_dim,
            context_size=args.context_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"HATA: {exc}")
        sys.exit(1)

    listened_idx = np.arange(args.base_size)
    candidate_idx = np.arange(args.base_size, len(df))

    user_embedding_profile = embeddings[listened_idx].mean(axis=0)
    user_feature_profile = features[listened_idx].mean(axis=0)
    listened_artists = listened_artist_set(df, listened_idx)

    embedding_scores = cosine_scores(embeddings[candidate_idx], user_embedding_profile)
    feature_scores = cosine_scores(features[candidate_idx], user_feature_profile)
    artist_scores = artist_overlap_score(df, candidate_idx, listened_artists)
    final_scores = 0.70 * embedding_scores + 0.20 * feature_scores + 0.10 * artist_scores

    recommendations = build_result_frame(
        df=df,
        candidate_idx=candidate_idx,
        top_k=args.top_k,
        score_columns={
            "embedding_similarity": embedding_scores,
            "feature_similarity": feature_scores,
            "artist_similarity": artist_scores,
            "final_score": final_scores,
        },
    )

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_csv = os.path.join(base_dir, f"{base_name}_pytorch_embedding_base{args.base_size}.csv")
    recommendations.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Veri dosyasi: {file_path}")
    print(f"Dinlenmis ilk sarki sayisi: {args.base_size}")
    print(f"Oneri sayisi: {args.top_k}")
    print(f"Kayit dosyasi: {out_csv}\n")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
