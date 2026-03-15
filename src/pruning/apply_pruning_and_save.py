import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_model_and_processor(
    model_type: str,
    model_path: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    dtype = dtype_map[torch_dtype]

    if model_type == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_path)

    elif model_type == "llava":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, processor


def get_transformer_layers(model, model_type: str):
    if model_type == "qwen":
        return model.model.layers
    elif model_type == "llava":
        return model.language_model.layers
    raise ValueError(f"Unsupported model_type: {model_type}")


def parse_predicted_keep_ratios(prediction_json_path: str, index: int = 0) -> List[float]:
    """
    Preferred field:
      - predicted_layer_keep_ratios

    Backward compatibility:
      - predicted_layer_ratios (treated as keep ratios in older local scripts)
    """
    data = load_json(prediction_json_path)
    if not isinstance(data, list):
        raise ValueError("Prediction JSON must be a list.")

    item = data[index]

    keep_ratios = item.get("predicted_layer_keep_ratios")
    if keep_ratios is None:
        keep_ratios = item.get("predicted_layer_ratios")

    if keep_ratios is None:
        raise ValueError(
            "Cannot find predicted layer keep ratios. Expected "
            "'predicted_layer_keep_ratios' or backward-compatible 'predicted_layer_ratios'."
        )

    return [float(x) for x in keep_ratios]


def importance_to_masks(importance_matrix: np.ndarray, keep_ratios: List[float]) -> List[torch.Tensor]:
    assert len(keep_ratios) == importance_matrix.shape[0], \
        "Number of keep ratios must match number of layers in importance matrix."

    masks = []
    for layer_idx in range(importance_matrix.shape[0]):
        layer_scores = torch.from_numpy(importance_matrix[layer_idx])
        keep_ratio = keep_ratios[layer_idx]

        num_neurons = layer_scores.shape[0]
        num_keep = max(1, int(num_neurons * keep_ratio))

        _, topk_indices = torch.topk(layer_scores, num_keep)

        mask = torch.zeros(num_neurons, dtype=torch.bool)
        mask[topk_indices] = True
        masks.append(mask)

    return masks


def apply_masks_to_model(model, model_type: str, masks: List[torch.Tensor]):
    layers = get_transformer_layers(model, model_type)

    for layer_idx, layer in enumerate(layers):
        mlp = layer.mlp
        mask = masks[layer_idx].to(mlp.gate_proj.weight.device)

        # gate_proj / up_proj: mask rows
        mlp.gate_proj.weight.data *= mask.unsqueeze(1)
        mlp.up_proj.weight.data *= mask.unsqueeze(1)

        # down_proj: mask columns
        mlp.down_proj.weight.data *= mask.unsqueeze(0)


def save_pruned_model_and_metadata(
    model,
    processor,
    save_dir: str,
    model_type: str,
    prediction_json_path: str,
    prediction_index: int,
    importance_path: str,
    keep_ratios: List[float],
):
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    metadata = {
        "model_type": model_type,
        "prediction_source": os.path.basename(prediction_json_path),
        "prediction_index": prediction_index,
        "importance_path": importance_path,
        "layer_keep_ratios": [round(float(x), 6) for x in keep_ratios],
        "layer_pruning_ratios": [round(1.0 - float(x), 6) for x in keep_ratios],
        "mean_keep_ratio": round(float(sum(keep_ratios) / len(keep_ratios)), 6),
        "mean_pruning_ratio": round(float(1.0 - sum(keep_ratios) / len(keep_ratios)), 6),
    }

    save_json(metadata, os.path.join(save_dir, "pruning_metadata.json"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply LOP-predicted layer ratios to a model using neuron importance and save the pruned model."
    )

    parser.add_argument("--model_type", type=str, choices=["qwen", "llava"], required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--importance_path", type=str, required=True)
    parser.add_argument("--prediction_json_path", type=str, required=True)
    parser.add_argument("--prediction_index", type=int, default=0)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    return parser.parse_args()


def main():
    args = parse_args()

    model, processor = build_model_and_processor(
        model_type=args.model_type,
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    keep_ratios = parse_predicted_keep_ratios(
        prediction_json_path=args.prediction_json_path,
        index=args.prediction_index,
    )
    importance_matrix = np.load(args.importance_path)

    masks = importance_to_masks(importance_matrix, keep_ratios)
    apply_masks_to_model(model, args.model_type, masks)

    save_pruned_model_and_metadata(
        model=model,
        processor=processor,
        save_dir=args.save_dir,
        model_type=args.model_type,
        prediction_json_path=args.prediction_json_path,
        prediction_index=args.prediction_index,
        importance_path=args.importance_path,
        keep_ratios=keep_ratios,
    )

    print(f"Saved pruned model to: {args.save_dir}")


if __name__ == "__main__":
    main()