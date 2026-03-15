import argparse
import json
import os
from typing import Any, Dict, List


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sorted_json_files(input_dir: str) -> List[str]:
    files = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.endswith(".json")
    ]
    return sorted(files)


def parse_mcts_result(data: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """
    Parse a single MCTS result file and convert it to one training sample for LOP.

    Expected preferred fields from the open-source MCTS output:
      - achieved_average_keep_ratio
      - layer_keep_ratios
      - layer_pruning_ratios
      - performance_metrics.accuracy
      - model_type

    Also supports older local-format fields for backward compatibility.
    """

    # Preferred fields from the cleaned MCTS output
    global_ratio = data.get("achieved_average_keep_ratio", data.get("achieved_keep_ratio"))
    if global_ratio is None:
        raise ValueError("Missing achieved keep ratio field.")

    layer_keep_ratios_dict = data.get("layer_keep_ratios")
    old_layer_ratios_dict = data.get("layer_ratios")

    # Backward compatibility:
    # old local script used "layer_ratios" to mean keep ratios
    if layer_keep_ratios_dict is None and old_layer_ratios_dict is not None:
        layer_keep_ratios_dict = old_layer_ratios_dict

    if layer_keep_ratios_dict is None:
        raise ValueError("Missing layer_keep_ratios / layer_ratios field.")

    # Sort by layer index
    layer_indices = sorted(int(k) for k in layer_keep_ratios_dict.keys())
    layer_keep_ratios = [float(layer_keep_ratios_dict[str(i)]) for i in layer_indices]
    layer_pruning_ratios = [1.0 - r for r in layer_keep_ratios]

    sample = {
        "global_keep_ratio": float(global_ratio),
        "global_pruning_ratio": 1.0 - float(global_ratio),
        "layer_keep_ratios": layer_keep_ratios,
        "layer_pruning_ratios": layer_pruning_ratios,
        "num_layers": len(layer_keep_ratios),
        "source_file": os.path.basename(source_file),
    }

    if "target_average_keep_ratio" in data:
        sample["target_keep_ratio"] = float(data["target_average_keep_ratio"])

    if "target_average_pruning_ratio" in data:
        sample["target_pruning_ratio"] = float(data["target_average_pruning_ratio"])

    if "model_type" in data:
        sample["model_type"] = data["model_type"]

    if "performance_metrics" in data and isinstance(data["performance_metrics"], dict):
        if "accuracy" in data["performance_metrics"]:
            sample["accuracy"] = float(data["performance_metrics"]["accuracy"])

    return sample


def build_training_dataset(input_dir: str) -> List[Dict[str, Any]]:
    processed = []
    skipped = []

    for file_path in sorted_json_files(input_dir):
        try:
            data = load_json(file_path)
            sample = parse_mcts_result(data, file_path)
            processed.append(sample)
        except Exception as e:
            skipped.append((os.path.basename(file_path), str(e)))

    print(f"Processed {len(processed)} files.")
    if skipped:
        print(f"Skipped {len(skipped)} files:")
        for filename, error in skipped:
            print(f"  - {filename}: {error}")

    return processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a folder of MCTS search results into LOP training data."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing MCTS result JSON files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged training dataset JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = build_training_dataset(args.input_dir)
    save_json(dataset, args.output_path)
    print(f"Saved merged training data to: {args.output_path}")


if __name__ == "__main__":
    main()