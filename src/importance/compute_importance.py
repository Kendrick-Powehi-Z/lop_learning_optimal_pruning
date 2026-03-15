import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.activations import ACT2FN

from qwen_vl_utils import process_vision_info


@dataclass
class ModelAdapter:
    model_type: str
    num_hidden_layers: int
    intermediate_size: int
    hidden_size: int
    hidden_act: str


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_image_path(record: Dict, image_root: str) -> str:
    if "image_path" in record and record["image_path"]:
        if os.path.isabs(record["image_path"]):
            return record["image_path"]
        return os.path.join(image_root, record["image_path"])

    if "index" in record:
        return os.path.join(image_root, f'{record["index"]}.png')

    raise ValueError("Cannot resolve image path from record. Expected 'image_path' or 'index'.")


def normalize_record(record: Dict, image_root: str) -> Dict:
    """
    Normalize dataset fields into a unified format:
    {
        "image_path": str,
        "text": str,
        ... (other metadata)
    }
    """
    image_path = resolve_image_path(record, image_root)

    text = (
        record.get("query")
        or record.get("input")
        or record.get("question")
        or ""
    )

    if not text:
        raise ValueError("Cannot find text field in record. Expected one of: query/input/question.")

    normalized = dict(record)
    normalized["image_path"] = image_path
    normalized["text"] = text
    return normalized


class InstrumentedMLP(nn.Module):
    """
    A drop-in replacement for the original MLP layer that keeps the same forward behavior
    while exposing per-neuron normalized activation statistics through `current_norms`.
    """

    def __init__(self, config, k_factor: float = 1.0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        self.k_factor = k_factor
        self.current_norms = None

    def forward(self, x):
        int_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        if x.shape[1] > 1:
            # Per-sample neuron importance statistics:
            # normalize across hidden dimension, then aggregate across sequence dimension.
            denom = int_states.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_states = int_states / denom
            self.current_norms = normalized_states.norm(dim=1)  # shape: (B, D)
        else:
            self.current_norms = None

        return self.down_proj(int_states)


def inject_instrumented_mlp_layers(model, model_type: str, k_schedule: List[float]):
    """
    Replace the original MLP blocks with InstrumentedMLP blocks so we can record
    neuron activation statistics during generation.
    """
    if model_type == "qwen":
        text_config = model.config
        layers = model.model.layers
    elif model_type == "llava":
        text_config = model.config.text_config
        layers = model.language_model.layers
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    for i, layer in enumerate(layers):
        new_mlp = InstrumentedMLP(text_config, k_schedule[i])

        # copy original weights
        new_mlp.gate_proj = layer.mlp.gate_proj
        new_mlp.up_proj = layer.mlp.up_proj
        new_mlp.down_proj = layer.mlp.down_proj
        new_mlp.act_fn = layer.mlp.act_fn

        layer.mlp = new_mlp

    return model


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
        adapter = ModelAdapter(
            model_type="qwen",
            num_hidden_layers=model.config.num_hidden_layers,
            intermediate_size=model.config.intermediate_size,
            hidden_size=model.config.hidden_size,
            hidden_act=model.config.hidden_act,
        )

    elif model_type == "llava":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        adapter = ModelAdapter(
            model_type="llava",
            num_hidden_layers=model.config.text_config.num_hidden_layers,
            intermediate_size=model.config.text_config.intermediate_size,
            hidden_size=model.config.text_config.hidden_size,
            hidden_act=model.config.text_config.hidden_act,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, processor, adapter


def get_transformer_layers(model, model_type: str):
    if model_type == "qwen":
        return model.model.layers
    elif model_type == "llava":
        return model.language_model.layers
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_messages(sample: Dict) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image_path"]},
                {"type": "text", "text": sample["text"]},
            ],
        }
    ]


@torch.no_grad()
def compute_neuron_importance(
    model,
    processor,
    model_type: str,
    samples: List[Dict],
    num_hidden_layers: int,
    intermediate_size: int,
    max_new_tokens: int = 128,
    save_predictions: bool = False,
):
    device = next(model.parameters()).device
    layers = get_transformer_layers(model, model_type)

    global_sum = torch.zeros(num_hidden_layers, intermediate_size, device=device)
    global_count = torch.zeros(num_hidden_layers, intermediate_size, device=device)

    predictions = []
    hooks = []

    def create_hook(layer_idx):
        def hook(module, _, __):
            if module.current_norms is not None:
                batch_sum = module.current_norms.sum(dim=0).detach().to(device)
                batch_size = module.current_norms.shape[0]
                global_sum[layer_idx] += batch_sum
                global_count[layer_idx] += batch_size
        return hook

    for layer_idx, layer in enumerate(layers):
        hooks.append(layer.mlp.register_forward_hook(create_hook(layer_idx)))

    for sample in tqdm(samples, desc="Computing neuron importance", unit="sample"):
        messages = build_messages(sample)

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if save_predictions:
            predictions.append({
                **sample,
                "prediction": output_text[0],
            })

    for hook in hooks:
        hook.remove()

    avg_norms = (global_sum / global_count.clamp(min=1)).detach().cpu().numpy()
    return avg_norms, predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FFN neuron importance for MLLMs.")

    parser.add_argument("--model_type", type=str, choices=["qwen", "llava"], required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--output_importance", type=str, required=True)
    parser.add_argument("--output_predictions", type=str, default=None)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    return parser.parse_args()


def main():
    args = parse_args()

    model, processor, adapter = build_model_and_processor(
        model_type=args.model_type,
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    k_schedule = [1.0 for _ in range(adapter.num_hidden_layers)]
    model = inject_instrumented_mlp_layers(model, args.model_type, k_schedule)

    raw_data = load_json(args.input_json)
    samples = [normalize_record(record, args.image_root) for record in raw_data]

    avg_norms, predictions = compute_neuron_importance(
        model=model,
        processor=processor,
        model_type=args.model_type,
        samples=samples,
        num_hidden_layers=adapter.num_hidden_layers,
        intermediate_size=adapter.intermediate_size,
        max_new_tokens=args.max_new_tokens,
        save_predictions=args.output_predictions is not None,
    )

    os.makedirs(os.path.dirname(args.output_importance), exist_ok=True)
    np.save(args.output_importance, avg_norms)

    if args.output_predictions is not None:
        save_json(predictions, args.output_predictions)

    print(f"Saved neuron importance to: {args.output_importance}")
    if args.output_predictions is not None:
        print(f"Saved predictions to: {args.output_predictions}")


if __name__ == "__main__":
    main()