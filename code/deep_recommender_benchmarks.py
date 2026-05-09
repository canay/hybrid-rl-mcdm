"""
CUDA-aware deep recommender baselines for the Hybrid RL-TOPSIS McAuley Amazon branch.

Models:
- BPR-MF
- NeuMF-style MLP fusion
- LightGCN
- SASRec-style sequential transformer

The script reads the processed McAuley Home & Kitchen data produced by
mccauley_home_data.py and evaluates against each user's temporal holdout set.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_mccauley_home" / "items.csv"
USERS_PATH = PROJECT_ROOT / "data" / "processed" / "amazon_mccauley_home" / "users.json"
OUT_JSON = PROJECT_ROOT / "results" / "deep_recommender_benchmarks.json"
OUT_CSV = PROJECT_ROOT / "results" / "deep_recommender_benchmarks_summary.csv"


def load_data(max_users: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    if not ITEMS_PATH.exists() or not USERS_PATH.exists():
        raise FileNotFoundError("Run scripts\\run_mccauley_home.ps1 or mccauley_home_data.py first.")
    items = pd.read_csv(ITEMS_PATH)
    with USERS_PATH.open("r", encoding="utf-8") as handle:
        users = json.load(handle)
    if max_users is not None:
        users = users[:max_users]
    return items, users


def build_indices(items: pd.DataFrame, users: list[dict]) -> tuple[dict[str, int], list[list[int]], list[set[int]]]:
    asin_to_idx = {str(asin): idx for idx, asin in enumerate(items["asin"].astype(str))}
    train_sequences: list[list[int]] = []
    test_sets: list[set[int]] = []
    for payload in users:
        train = [asin_to_idx[row["asin"]] for row in payload["train"] if row["asin"] in asin_to_idx]
        test = {asin_to_idx[row["asin"]] for row in payload["test"] if row["asin"] in asin_to_idx}
        if train and test:
            train_sequences.append(train)
            test_sets.append(test)
    return asin_to_idx, train_sequences, test_sets


class PairDataset(Dataset):
    def __init__(self, train_sequences: list[list[int]], n_items: int, seed: int) -> None:
        self.pairs = [(u, item) for u, seq in enumerate(train_sequences) for item in seq]
        self.user_pos = [set(seq) for seq in train_sequences]
        self.n_items = n_items
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        u, pos = self.pairs[idx]
        neg = int(self.rng.randint(0, self.n_items))
        while neg in self.user_pos[u]:
            neg = int(self.rng.randint(0, self.n_items))
        return int(u), int(pos), int(neg)


class SeqDataset(Dataset):
    def __init__(self, train_sequences: list[list[int]], n_items: int, max_len: int, seed: int) -> None:
        self.samples = []
        for u, seq in enumerate(train_sequences):
            for t in range(1, len(seq)):
                self.samples.append((u, seq[max(0, t - max_len):t], seq[t]))
        self.user_pos = [set(seq) for seq in train_sequences]
        self.n_items = n_items
        self.max_len = max_len
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, int]:
        u, prefix, pos = self.samples[idx]
        padded = [0] * (self.max_len - len(prefix)) + [item + 1 for item in prefix[-self.max_len:]]
        neg = int(self.rng.randint(0, self.n_items))
        while neg in self.user_pos[u]:
            neg = int(self.rng.randint(0, self.n_items))
        return torch.tensor(padded, dtype=torch.long), int(u), int(pos), int(neg)


class MF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, factors)
        self.item = nn.Embedding(n_items, factors)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(dim=-1)

    def full_scores(self) -> torch.Tensor:
        return self.user.weight @ self.item.weight.T


class NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int) -> None:
        super().__init__()
        self.user_gmf = nn.Embedding(n_users, factors)
        self.item_gmf = nn.Embedding(n_items, factors)
        self.user_mlp = nn.Embedding(n_users, factors)
        self.item_mlp = nn.Embedding(n_items, factors)
        self.mlp = nn.Sequential(
            nn.Linear(2 * factors, 2 * factors),
            nn.ReLU(),
            nn.Linear(2 * factors, factors),
            nn.ReLU(),
        )
        self.out = nn.Linear(2 * factors, 1)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.user_gmf(users) * self.item_gmf(items)
        mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    def full_scores(self, batch_users: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        rows = []
        for users in batch_users.split(128):
            u = users[:, None].expand(-1, len(item_ids)).reshape(-1)
            i = item_ids[None, :].expand(len(users), -1).reshape(-1)
            rows.append(self.score(u, i).reshape(len(users), len(item_ids)))
        return torch.cat(rows, dim=0)


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int, norm_adj: torch.Tensor, layers: int) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.layers = layers
        self.user = nn.Embedding(n_users, factors)
        self.item = nn.Embedding(n_items, factors)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        emb0 = torch.cat([self.user.weight, self.item.weight], dim=0)
        embs = [emb0]
        emb = emb0
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            embs.append(emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.embeddings()
        return (user_emb[users] * item_emb[items]).sum(dim=-1)

    def full_scores(self) -> torch.Tensor:
        user_emb, item_emb = self.embeddings()
        return user_emb @ item_emb.T


class SASRec(nn.Module):
    def __init__(self, n_items: int, factors: int, max_len: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.max_len = max_len
        self.item = nn.Embedding(n_items + 1, factors, padding_idx=0)
        self.pos = nn.Embedding(max_len, factors)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=factors,
            nhead=heads,
            dim_feedforward=4 * factors,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(factors)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(self.max_len, device=seq.device).unsqueeze(0)
        x = self.item(seq) + self.pos(positions)
        causal = torch.triu(torch.ones(self.max_len, self.max_len, device=seq.device), diagonal=1).bool()
        pad = seq.eq(0)
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        lengths = torch.clamp((~pad).sum(dim=1) - 1, min=0)
        return self.norm(h[torch.arange(seq.size(0), device=seq.device), lengths])

    def score(self, seq: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        h = self.encode(seq)
        return (h * self.item(items + 1)).sum(dim=-1)

    def full_scores(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.encode(seq)
        return h @ self.item.weight[1:].T


def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


def build_lightgcn_adj(train_sequences: list[list[int]], n_items: int, device: torch.device) -> torch.Tensor:
    n_users = len(train_sequences)
    rows = []
    cols = []
    for u, seq in enumerate(train_sequences):
        for item in set(seq):
            rows.extend([u, n_users + item])
            cols.extend([n_users + item, u])
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(rows), dtype=torch.float32)
    n_nodes = n_users + n_items
    deg = torch.zeros(n_nodes, dtype=torch.float32)
    deg.scatter_add_(0, idx[0], vals)
    norm_vals = vals / torch.sqrt(deg[idx[0]].clamp_min(1.0) * deg[idx[1]].clamp_min(1.0))
    return torch.sparse_coo_tensor(idx, norm_vals, (n_nodes, n_nodes)).coalesce().to(device)


def train_pairwise(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int, lr: float, weight_decay: float) -> float:
    start = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for users, pos, neg in loader:
            users = users.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            opt.zero_grad(set_to_none=True)
            loss = bpr_loss(model.score(users, pos), model.score(users, neg))
            loss.backward()
            opt.step()
    return time.perf_counter() - start


def train_sasrec(model: SASRec, loader: DataLoader, device: torch.device, epochs: int, lr: float, weight_decay: float) -> float:
    start = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        for seq, _users, pos, neg in loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            opt.zero_grad(set_to_none=True)
            loss = bpr_loss(model.score(seq, pos), model.score(seq, neg))
            loss.backward()
            opt.step()
    return time.perf_counter() - start


def metrics_from_scores(scores: torch.Tensor, train_sequences: list[list[int]], test_sets: list[set[int]], k: int) -> dict:
    scores = scores.detach().cpu().numpy()
    hit = []
    ndcg = []
    mrr = []
    coverage = set()
    for u, row in enumerate(scores):
        row = row.copy()
        row[list(set(train_sequences[u]))] = -np.inf
        ranked = np.argsort(row)[::-1][:k].tolist()
        coverage.update(ranked)
        truth = test_sets[u]
        ranks = [rank + 1 for rank, item in enumerate(ranked) if item in truth]
        if ranks:
            best_rank = min(ranks)
            hit.append(1.0)
            ndcg.append(1.0 / math.log2(best_rank + 1.0))
            mrr.append(1.0 / best_rank)
        else:
            hit.append(0.0)
            ndcg.append(0.0)
            mrr.append(0.0)
    return {
        f"hit_at_{k}": float(np.mean(hit)),
        f"ndcg_at_{k}": float(np.mean(ndcg)),
        f"mrr_at_{k}": float(np.mean(mrr)),
        f"coverage_at_{k}": float(len(coverage) / scores.shape[1]),
    }


def seq_tensor(train_sequences: list[list[int]], max_len: int, device: torch.device) -> torch.Tensor:
    rows = []
    for seq in train_sequences:
        clipped = seq[-max_len:]
        rows.append([0] * (max_len - len(clipped)) + [item + 1 for item in clipped])
    return torch.tensor(rows, dtype=torch.long, device=device)


def summarize(raw: dict[str, list[float]]) -> dict:
    out = {}
    for key, values in raw.items():
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "raw": [float(x) for x in arr],
        }
    return out


def run_once(
    train_sequences: list[list[int]],
    test_sets: list[set[int]],
    n_items: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = PairDataset(train_sequences, n_items, seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    seq_dataset = SeqDataset(train_sequences, n_items, args.max_len, seed + 300)
    seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    user_ids = torch.arange(len(train_sequences), device=device)

    results = {}

    mf = MF(len(train_sequences), n_items, args.factors).to(device)
    train_time = train_pairwise(mf, loader, device, args.epochs, args.lr, args.weight_decay)
    with torch.no_grad():
        scores = mf.full_scores()
    results["bpr_mf"] = {"train_seconds": train_time, **metrics_from_scores(scores, train_sequences, test_sets, args.k)}

    neumf = NeuMF(len(train_sequences), n_items, args.factors).to(device)
    train_time = train_pairwise(neumf, loader, device, args.epochs, args.lr, args.weight_decay)
    item_ids = torch.arange(n_items, device=device)
    with torch.no_grad():
        scores = neumf.full_scores(user_ids, item_ids)
    results["neumf"] = {"train_seconds": train_time, **metrics_from_scores(scores, train_sequences, test_sets, args.k)}

    adj = build_lightgcn_adj(train_sequences, n_items, device)
    lgcn = LightGCN(len(train_sequences), n_items, args.factors, adj, args.gcn_layers).to(device)
    train_time = train_pairwise(lgcn, loader, device, args.epochs, args.lr, args.weight_decay)
    with torch.no_grad():
        scores = lgcn.full_scores()
    results["lightgcn"] = {"train_seconds": train_time, **metrics_from_scores(scores, train_sequences, test_sets, args.k)}

    sasrec = SASRec(n_items, args.factors, args.max_len, args.heads, args.sasrec_layers, args.dropout).to(device)
    train_time = train_sasrec(sasrec, seq_loader, device, args.epochs, args.lr, args.weight_decay)
    full_seq = seq_tensor(train_sequences, args.max_len, device)
    with torch.no_grad():
        scores = sasrec.full_scores(full_seq)
    results["sasrec"] = {"train_seconds": train_time, **metrics_from_scores(scores, train_sequences, test_sets, args.k)}

    return results


def run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    items, users = load_data(max_users=args.max_users)
    _asin_to_idx, train_sequences, test_sets = build_indices(items, users)
    n_items = len(items)

    raw: dict[str, dict[str, list[float]]] = {}
    print("=" * 72)
    print("Hybrid RL-TOPSIS deep recommender benchmarks on McAuley Amazon Home")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Users: {len(train_sequences)}; Items: {n_items}; Runs: {args.runs}; Epochs: {args.epochs}")
    print("-" * 72)

    started = time.perf_counter()
    for run_idx in range(args.runs):
        result = run_once(train_sequences, test_sets, n_items, 8800 + run_idx * 31, args, device)
        for model, metrics in result.items():
            raw.setdefault(model, {})
            for key, value in metrics.items():
                raw[model].setdefault(key, []).append(float(value))
        best = max(result, key=lambda name: result[name][f"ndcg_at_{args.k}"])
        print(
            f"Run {run_idx + 1}/{args.runs}: "
            f"best={best}, ndcg@{args.k}={result[best][f'ndcg_at_{args.k}']:.4f}"
        )

    return {
        "config": {
            "dataset": "McAuley/UCSD Amazon Home and Kitchen 5-core processed for Hybrid RL-TOPSIS",
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "runs": int(args.runs),
            "epochs": int(args.epochs),
            "factors": int(args.factors),
            "batch_size": int(args.batch_size),
            "k": int(args.k),
            "max_len": int(args.max_len),
            "n_users": int(len(train_sequences)),
            "n_items": int(n_items),
        },
        "summary": {model: summarize(metrics) for model, metrics in raw.items()},
        "total_runtime_seconds": float(time.perf_counter() - started),
    }


def save(report: dict, k: int) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    rows = []
    metric = f"ndcg_at_{k}"
    for model, metrics in report["summary"].items():
        row = {"model": model}
        for key, stats in metrics.items():
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
        rows.append(row)
    pd.DataFrame(rows).sort_values(f"{metric}_mean", ascending=False).to_csv(OUT_CSV, index=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA-aware deep recommender baselines.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--gcn-layers", type=int, default=2)
    parser.add_argument("--sasrec-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-cuda", dest="cuda", action="store_false")
    parser.set_defaults(cuda=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    save(report, args.k)
    print("=" * 72)
    print("Deep benchmark summary")
    print("=" * 72)
    metric = f"ndcg_at_{args.k}"
    for model, metrics in sorted(report["summary"].items(), key=lambda item: item[1][metric]["mean"], reverse=True):
        print(
            f"{model:10s}: "
            f"Hit@{args.k}={metrics[f'hit_at_{args.k}']['mean']:.4f}, "
            f"NDCG@{args.k}={metrics[metric]['mean']:.4f}, "
            f"MRR@{args.k}={metrics[f'mrr_at_{args.k}']['mean']:.4f}, "
            f"train_s={metrics['train_seconds']['mean']:.2f}"
        )
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()

