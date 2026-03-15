import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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
        if global_keep_ratio.ndim == 1:
            global_keep_ratio = global_keep_ratio.unsqueeze(1)

        h0 = self.global_embed(global_keep_ratio)  # (B, d_model)
        x = h0.unsqueeze(1) + self.pos_embedding.unsqueeze(0)  # (B, L, d_model)
        x = self.transformer(x)  # (B, L, d_model)
        out = self.output_proj(x).squeeze(-1)  # (B, L)
        return out


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        # backward compatibility: older checkpoint may directly be state_dict
        return {
            "model_state_dict": checkpoint,
            "train_config": {
                "d_model": 128,
                "num_transformer_layers": 4,
                "num_heads": 4,
            },
            "num_output_layers": 36,
        }

    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict[str, Any], device: str):
    train_config = checkpoint.get("train_config", {})
    num_output_layers = checkpoint.get("num_output_layers", 36)

    model = LOPRatioPredictor(
        d_model=train_config.get("d_model", 128),
        num_transformer_layers=train_config.get("num_transformer_layers", 4),
        num_heads=train_config.get("num_heads", 4),
        num_output_layers=num_output_layers,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    global_keep_ratios: List[float],
    device: str,
) -> List[Dict[str, Any]]:
    input_tensor = torch.tensor(global_keep_ratios, dtype=torch.float32, device=device)
    pred = model(input_tensor)  # (B, L)

    results = []
    for i, keep_ratio in enumerate(global_keep_ratios):
        layer_keep_ratios = pred[i].detach().cpu().tolist()
        layer_pruning_ratios = [1.0 - x for x in layer_keep_ratios]

        mean_keep_ratio = sum(layer_keep_ratios) / len(layer_keep_ratios)
        mean_pruning_ratio = 1.0 - mean_keep_ratio

        result = {
            "global_keep_ratio": round(float(keep_ratio), 6),
            "global_pruning_ratio": round(1.0 - float(keep_ratio), 6),
            "predicted_layer_keep_ratios": [round(float(x), 6) for x in layer_keep_ratios],
            "predicted_layer_pruning_ratios": [round(float(x), 6) for x in layer_pruning_ratios],
            "mean_predicted_keep_ratio": round(float(mean_keep_ratio), 6),
            "mean_predicted_pruning_ratio": round(float(mean_pruning_ratio), 6),
        }
        results.append(result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained LOP ratio predictor.")

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint_best.pt or checkpoint_final.pt",
    )
    parser.add_argument(
        "--global_keep_ratios",
        type=float,
        nargs="+",
        required=True,
        help="One or more target global keep ratios for inference.",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="Path to save predicted layer ratios.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    start_time = time.time()

    checkpoint = load_checkpoint(args.checkpoint_path, args.device)
    model = build_model_from_checkpoint(checkpoint, args.device)

    results = run_inference(
        model=model,
        global_keep_ratios=args.global_keep_ratios,
        device=args.device,
    )

    save_json(results, args.output_json_path)

    for result in results:
        print(
            f'Global keep ratio: {result["global_keep_ratio"]:.4f} | '
            f'Mean predicted keep ratio: {result["mean_predicted_keep_ratio"]:.4f}'
        )

    elapsed_time = time.time() - start_time
    print(f"\nSaved predictions to: {args.output_json_path}")
    print(f"Inference finished in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()