import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


def extract_option(text: str) -> str:
    """
    Extract a normalized answer token from free-form model output.
    Supports multiple-choice answers (A-E) and yes/no answers.
    """
    if not isinstance(text, (str, bytes)):
        return "OTHER"

    text = text.strip().upper()

    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)

    for char in text:
        if char in "ABCDE":
            return char

    text_clean = text.lower().replace(".", "").strip()
    if text_clean in ["yes", "no"]:
        return text_clean.upper()
    if text_clean == "y":
        return "YES"
    if text_clean == "n":
        return "NO"
    if text_clean.startswith("yes"):
        return "YES"
    if text_clean.startswith("no"):
        return "NO"

    return "OTHER"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class SearchConfig:
    simulations: int = 300
    sample_size: int = 200
    delta_step: float = 0.1
    min_keep_ratio: float = 0.1
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    max_new_tokens: int = 32
    expansion_actions: int = 4
    expansion_visits_threshold: int = 3


@dataclass
class ModelAdapter:
    model_type: str
    num_hidden_layers: int


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
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, processor, adapter


def get_transformer_layers(model, model_type: str):
    if model_type == "qwen":
        return model.model.layers
    if model_type == "llava":
        return model.language_model.layers
    raise ValueError(f"Unsupported model_type: {model_type}")


def normalize_eval_record(record: Dict, image_root: str) -> Dict:
    image_path = record.get("image_path")
    if image_path is None and "index" in record:
        image_path = f'{record["index"]}.png'
    if image_path is None:
        raise ValueError("Cannot resolve image path from record.")

    if not os.path.isabs(image_path):
        image_path = os.path.join(image_root, image_path)

    input_text = (
        record.get("input_text")
        or record.get("input")
        or record.get("query")
        or record.get("question")
        or ""
    )
    if not input_text:
        raise ValueError("Cannot resolve input text from record.")

    answer = record.get("answer", "")
    return {
        "image_path": image_path,
        "input_text": input_text,
        "answer": answer,
    }


class GlobalPruningState:
    """
    State stores layer-wise KEEP ratios, not pruning ratios.
    Example:
        layer_keep_ratios[layer_id] = 0.7
    means 70% neurons are kept in that layer.
    """

    def __init__(self, layer_keep_ratios: Dict[int, float], layer_params: Dict[int, int]):
        self.layer_keep_ratios = layer_keep_ratios.copy()
        self.layer_params = layer_params
        self.total_params = sum(layer_params.values())

    def validate(self, max_average_keep_ratio: float) -> bool:
        avg_keep_ratio = sum(self.layer_keep_ratios.values()) / len(self.layer_keep_ratios)
        return avg_keep_ratio <= max_average_keep_ratio

    def current_pruning_ratio(self) -> float:
        kept = sum(
            ratio * params
            for ratio, params in zip(self.layer_keep_ratios.values(), self.layer_params.values())
        )
        return 1.0 - kept / self.total_params

    def generate_actions(
        self,
        num_actions: int,
        max_average_keep_ratio: float,
        depth: int,
        delta_step: float,
        min_keep_ratio: float,
    ) -> List["GlobalPruningState"]:
        valid_actions = []
        current_delta = max(delta_step * (0.9 ** depth), 0.01)

        for _ in range(num_actions * 2):
            new_ratios = self.layer_keep_ratios.copy()

            for lid in new_ratios:
                perturb = np.random.uniform(-current_delta, current_delta)
                new_ratios[lid] = float(np.clip(new_ratios[lid] + perturb, min_keep_ratio, 1.0))

            avg_keep_ratio = sum(new_ratios.values()) / len(new_ratios)
            if avg_keep_ratio > max_average_keep_ratio:
                scale = max_average_keep_ratio / avg_keep_ratio
                for lid in new_ratios:
                    new_ratios[lid] = float(np.clip(new_ratios[lid] * scale, min_keep_ratio, 1.0))

            new_state = GlobalPruningState(new_ratios, self.layer_params)
            if new_state.validate(max_average_keep_ratio):
                valid_actions.append(new_state)
                if len(valid_actions) >= num_actions:
                    break

        return valid_actions


class PruningEvaluator:
    """
    Evaluate a candidate layer-wise keep-ratio configuration by masking FFN neurons
    according to precomputed neuron importance scores.
    """

    def __init__(
        self,
        model,
        processor,
        model_type: str,
        dataset_path: str,
        image_root: str,
        sample_size: int,
        importance_scores: Optional[np.ndarray] = None,
        max_new_tokens: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.importance_scores = importance_scores
        self.max_new_tokens = max_new_tokens

        raw_data = load_json(dataset_path)
        self.dataset = [
            normalize_eval_record(record, image_root)
            for record in raw_data[:sample_size]
        ]

    def evaluate(self, state: GlobalPruningState) -> float:
        backup_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        layers = get_transformer_layers(self.model, self.model_type)

        try:
            for layer_id, keep_ratio in state.layer_keep_ratios.items():
                self._mask_ffn_layer(layers[layer_id].mlp, keep_ratio, layer_id)

            with torch.no_grad():
                accuracy = self._evaluate_accuracy(self.model)

        finally:
            self.model.load_state_dict(backup_state)
            torch.cuda.empty_cache()

        return accuracy

    def _mask_ffn_layer(self, ffn, keep_ratio: float, layer_id: int):
        """
        Mask neurons in-place using importance ranking while keeping tensor shapes unchanged.
        """
        intermediate_size = ffn.gate_proj.weight.shape[0]
        device = ffn.gate_proj.weight.device

        if self.importance_scores is not None:
            scores = torch.tensor(self.importance_scores[layer_id], device=device)
        else:
            scores = torch.rand(intermediate_size, device=device)

        num_keep = max(1, int(intermediate_size * keep_ratio))
        _, topk_indices = torch.topk(scores, num_keep, largest=True)

        mask = torch.zeros(intermediate_size, dtype=torch.bool, device=device)
        mask[topk_indices] = True

        gate_weight = ffn.gate_proj.weight.data
        up_weight = ffn.up_proj.weight.data
        down_weight = ffn.down_proj.weight.data

        binary_mask = mask.unsqueeze(1).to(gate_weight.dtype)
        binary_mask_down = mask.unsqueeze(0).to(down_weight.dtype)

        ffn.gate_proj.weight.data.copy_(gate_weight * binary_mask)
        ffn.up_proj.weight.data.copy_(up_weight * binary_mask)
        ffn.down_proj.weight.data.copy_(down_weight * binary_mask_down)

    @torch.no_grad()
    def _evaluate_accuracy(self, model) -> float:
        correct = 0

        for sample in self.dataset:
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": sample["input_text"]},
                    ],
                }]

                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, _ = process_vision_info(messages)

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                prediction = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )[0]

                pred_option = extract_option(prediction)
                answer_option = extract_option(sample["answer"])

                if pred_option.strip() == answer_option.strip():
                    correct += 1

            except Exception as e:
                print(f"Evaluation error: {str(e)}")

        return correct / len(self.dataset) if self.dataset else 0.0


class GlobalPruningMCTS:
    class MCTSNode:
        def __init__(self, state: GlobalPruningState, depth: int = 0):
            self.state = state
            self.children = []
            self.visits = 0
            self.total_reward = 0.0
            self.parent = None
            self.depth = depth

        def uct_value(self, parent_visits: int, exploration: float = 1.414) -> float:
            if self.visits == 0:
                return float("inf")
            return (
                (self.total_reward / self.visits)
                + exploration * math.sqrt(math.log(parent_visits) / self.visits)
            )

    def __init__(
        self,
        evaluator: PruningEvaluator,
        layer_params: Dict[int, int],
        max_average_keep_ratio: float,
        search_config: SearchConfig,
    ):
        self.evaluator = evaluator
        self.layer_params = layer_params
        self.max_average_keep_ratio = max_average_keep_ratio
        self.search_config = search_config

        init_keep_ratios = {lid: max_average_keep_ratio for lid in layer_params}
        self.root = self.MCTSNode(
            GlobalPruningState(init_keep_ratios, layer_params),
            depth=0,
        )

    def search(self) -> Dict[int, float]:
        for _ in tqdm(range(self.search_config.simulations), desc="MCTS search"):
            node = self._select()
            expanded = self._expand(node)
            reward = self._simulate(expanded)
            self._backpropagate(expanded, reward)

        return self._best_config()

    def _select(self):
        node = self.root
        while node.children:
            node = max(node.children, key=lambda n: n.uct_value(node.visits))
        return node

    def _expand(self, node):
        if node.visits < self.search_config.expansion_visits_threshold:
            for action_state in node.state.generate_actions(
                num_actions=self.search_config.expansion_actions,
                max_average_keep_ratio=self.max_average_keep_ratio,
                depth=node.depth,
                delta_step=self.search_config.delta_step,
                min_keep_ratio=self.search_config.min_keep_ratio,
            ):
                child = self.MCTSNode(action_state, depth=node.depth + 1)
                child.parent = node
                node.children.append(child)

        return random.choice(node.children) if node.children else node

    def _simulate(self, node):
        return self.evaluator.evaluate(node.state)

    def _backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def _best_config(self) -> Dict[int, float]:
        candidates = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            if node.state.validate(self.max_average_keep_ratio):
                candidates.append(node)
            stack.extend(node.children)

        candidates = [n for n in candidates if n.visits > 0]
        if not candidates:
            raise RuntimeError("No valid candidates were visited during MCTS search.")

        best_node = max(candidates, key=lambda n: n.total_reward / n.visits)
        return best_node.state.layer_keep_ratios

    def evaluate_best(self, best_keep_ratios: Dict[int, float]) -> float:
        state = GlobalPruningState(best_keep_ratios, self.layer_params)
        return self.evaluator.evaluate(state)


def build_layer_param_dict(model, model_type: str) -> Dict[int, int]:
    layers = get_transformer_layers(model, model_type)
    return {
        lid: layer.mlp.gate_proj.weight.numel() * 3
        for lid, layer in enumerate(layers)
    }


def save_search_result(
    output_path: str,
    model_type: str,
    target_keep_ratio: float,
    best_keep_ratios: Dict[int, float],
    accuracy: float,
):
    layer_pruning_ratios = {str(k): round(1.0 - v, 4) for k, v in best_keep_ratios.items()}
    layer_keep_ratios = {str(k): round(v, 4) for k, v in best_keep_ratios.items()}

    result = {
        "model_type": model_type,
        "target_average_keep_ratio": target_keep_ratio,
        "target_average_pruning_ratio": round(1.0 - target_keep_ratio, 4),
        "achieved_average_keep_ratio": round(sum(best_keep_ratios.values()) / len(best_keep_ratios), 4),
        "achieved_average_pruning_ratio": round(
            1.0 - (sum(best_keep_ratios.values()) / len(best_keep_ratios)),
            4,
        ),
        "layer_keep_ratios": layer_keep_ratios,
        "layer_pruning_ratios": layer_pruning_ratios,
        "performance_metrics": {
            "accuracy": round(float(accuracy), 6),
        },
    }

    save_json(result, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="MCTS search for layer-wise FFN pruning ratios.")

    parser.add_argument("--model_type", type=str, choices=["qwen", "llava"], required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--importance_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--target_keep_ratios", type=float, nargs="+", required=True)

    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--delta_step", type=float, default=0.1)
    parser.add_argument("--min_keep_ratio", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=32)

    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32", "auto"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    return parser.parse_args()


def main():
    args = parse_args()

    search_config = SearchConfig(
        simulations=args.simulations,
        sample_size=args.sample_size,
        delta_step=args.delta_step,
        min_keep_ratio=args.min_keep_ratio,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
    )

    model, processor, _ = build_model_and_processor(
        model_type=args.model_type,
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    importance_scores = np.load(args.importance_path)
    layer_params = build_layer_param_dict(model, args.model_type)

    evaluator = PruningEvaluator(
        model=model,
        processor=processor,
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        image_root=args.image_root,
        sample_size=args.sample_size,
        importance_scores=importance_scores,
        max_new_tokens=args.max_new_tokens,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for target_keep_ratio in args.target_keep_ratios:
        print(f"\nRunning MCTS search for target average keep ratio = {target_keep_ratio:.4f}")

        mcts = GlobalPruningMCTS(
            evaluator=evaluator,
            layer_params=layer_params,
            max_average_keep_ratio=target_keep_ratio,
            search_config=search_config,
        )

        best_keep_ratios = mcts.search()
        accuracy = mcts.evaluate_best(best_keep_ratios)

        output_name = f"mcts_keep_{target_keep_ratio:.4f}.json"
        output_path = os.path.join(args.output_dir, output_name)

        save_search_result(
            output_path=output_path,
            model_type=args.model_type,
            target_keep_ratio=target_keep_ratio,
            best_keep_ratios=best_keep_ratios,
            accuracy=accuracy,
        )

        print(f"Saved search result to: {output_path}")


if __name__ == "__main__":
    main()