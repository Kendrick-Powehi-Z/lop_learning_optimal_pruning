import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    d_model: int = 128
    num_transformer_layers: int = 4
    num_heads: int = 4
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "cuda"


class LOPRatioDataset(Dataset):
    """
    Training dataset for LOP predictor.

    Expected preferred fields in each sample:
      - global_keep_ratio
      - layer_keep_ratios

    Also supports older fields for backward compatibility:
      - global_ratio
      - layer_ratios
    """

    def __init__(self, json_path: str):
        self.data = load_json(json_path)

        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError(f"Dataset at {json_path} is empty or not a list.")

        self.samples = []
        expected_num_layers = None

        for sample in self.data:
            global_keep_ratio = sample.get("global_keep_ratio", sample.get("global_ratio"))
            layer_keep_ratios = sample.get("layer_keep_ratios", sample.get("layer_ratios"))

            if global_keep_ratio is None or layer_keep_ratios is None:
                continue

            if expected_num_layers is None:
                expected_num_layers = len(layer_keep_ratios)

            if len(layer_keep_ratios) != expected_num_layers:
                # skip inconsistent samples
                continue

            self.samples.append({
                "global_keep_ratio": float(global_keep_ratio),
                "layer_keep_ratios": [float(x) for x in layer_keep_ratios],
            })

        if len(self.samples) == 0:
            raise ValueError("No valid samples found in training dataset.")

        self.num_layers = len(self.samples[0]["layer_keep_ratios"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        global_keep_ratio = torch.tensor([sample["global_keep_ratio"]], dtype=torch.float32)
        layer_keep_ratios = torch.tensor(sample["layer_keep_ratios"], dtype=torch.float32)
        return global_keep_ratio, layer_keep_ratios


class LOPRatioPredictor(nn.Module):
    """
    Predictor that maps a global keep ratio to layer-wise keep ratios.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        num_output_layers: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_output_layers = num_output_layers

        self.global_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embedding = nn.Parameter(torch.randn(num_output_layers, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, global_keep_ratio: torch.Tensor) -> torch.Tensor:
        """
        global_keep_ratio: (B, 1) or (B,)
        returns: (B, num_output_layers)
        """
        if global_keep_ratio.ndim == 1:
            global_keep_ratio = global_keep_ratio.unsqueeze(1)

        batch_size = global_keep_ratio.size(0)
        h0 = self.global_embed(global_keep_ratio)  # (B, d_model)
        x = h0.unsqueeze(1) + self.pos_embedding.unsqueeze(0)  # (B, L, d_model)
        x = self.transformer(x)  # (B, L, d_model)
        out = self.output_proj(x).squeeze(-1)  # (B, L)
        return out


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for global_keep_ratio, target_layer_keep_ratios in dataloader:
            global_keep_ratio = global_keep_ratio.to(device)
            target_layer_keep_ratios = target_layer_keep_ratios.to(device)

            pred = model(global_keep_ratio)
            loss = criterion(pred, target_layer_keep_ratios)

            batch_size = global_keep_ratio.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for global_keep_ratio, target_layer_keep_ratios in tqdm(dataloader, desc="Training", leave=False):
        global_keep_ratio = global_keep_ratio.to(device)
        target_layer_keep_ratios = target_layer_keep_ratios.to(device)

        optimizer.zero_grad()
        pred = model(global_keep_ratio)
        loss = criterion(pred, target_layer_keep_ratios)
        loss.backward()
        optimizer.step()

        batch_size = global_keep_ratio.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def save_checkpoint(path: str, model, config: TrainConfig, num_output_layers: int, history: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": asdict(config),
            "num_output_layers": num_output_layers,
            "history": history,
        },
        path,
    )


def train(
    json_path: str,
    output_dir: str,
    config: TrainConfig,
):
    set_seed(config.seed)

    dataset = LOPRatioDataset(json_path)
    num_output_layers = dataset.num_layers

    val_size = max(1, int(len(dataset) * config.val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if val_dataset else None

    model = LOPRatioPredictor(
        d_model=config.d_model,
        num_transformer_layers=config.num_transformer_layers,
        num_heads=config.num_heads,
        num_output_layers=num_output_layers,
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = []
    best_val_loss = math.inf
    best_ckpt_path = os.path.join(output_dir, "checkpoint_best.pt")
    final_ckpt_path = os.path.join(output_dir, "checkpoint_final.pt")
    log_path = os.path.join(output_dir, "train_log.json")

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, config.device)
        else:
            val_loss = train_loss

        record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        history.append(record)

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_ckpt_path,
                model=model,
                config=config,
                num_output_layers=num_output_layers,
                history=history,
            )

    save_checkpoint(
        final_ckpt_path,
        model=model,
        config=config,
        num_output_layers=num_output_layers,
        history=history,
    )

    save_json(
        {
            "dataset_path": json_path,
            "num_samples": len(dataset),
            "num_output_layers": num_output_layers,
            "best_val_loss": float(best_val_loss),
            "history": history,
            "train_config": asdict(config),
        },
        log_path,
    )

    print(f"Saved best checkpoint to: {best_ckpt_path}")
    print(f"Saved final checkpoint to: {final_ckpt_path}")
    print(f"Saved training log to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LOP ratio predictor.")

    parser.add_argument("--json_path", type=str, required=True, help="Path to the merged LOP training dataset JSON.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
    )

    train(
        json_path=args.json_path,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()