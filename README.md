# LOP: Learning Optimal Pruning for Efficient On-Demand MLLMs Scaling

Official code for our paper on **LOP**, a lightweight predictor for efficient **on-demand structured pruning** of multimodal large language models (MLLMs).

This repository provides the core research code used in our paper, including:

- **FFN neuron importance estimation**
- **MCTS-based search** for high-quality layer-wise pruning configurations
- **LOP predictor training**
- **LOP inference**
- **Applying predicted pruning ratios and saving pruned models**

This release is intended for **paper reproduction and research use**, rather than a fully engineered deployment toolkit.

---

## Overview

The overall pipeline of LOP consists of the following stages:

1. **Compute neuron importance** on a calibration dataset  
2. **Run MCTS search** to obtain high-quality layer-wise FFN keep ratios under target global constraints  
3. **Prepare training data** from MCTS search results  
4. **Train the LOP predictor** to map a global ratio to layer-wise ratios  
5. **Run inference** with the trained predictor  
6. **Apply the predicted pruning ratios** and save the pruned model  

---

## Supported Models

The current code supports the following multimodal LLMs:

- **Qwen2.5-VL-7B**
- **LLaVA-OneVision-1.5-4B / 8B**  
  (depending on the checkpoint you use)

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── src/
│   ├── importance/
│   │   └── compute_importance.py
│   ├── search/
│   │   └── mcts_search.py
│   ├── lop/
│   │   ├── prepare_training_data.py
│   │   ├── train.py
│   │   └── infer.py
│   └── pruning/
│       └── apply_pruning_and_save.py
├── scripts/
│   ├── run_importance.sh
│   ├── run_mcts_search.sh
│   ├── run_prepare_lop_data.sh
│   ├── run_train_lop.sh
│   ├── run_infer_lop.sh
│   └── run_apply_pruning.sh
└── outputs/
    ├── importance/
    ├── mcts/
    ├── lop_data/
    ├── lop_predictor/
    ├── pruning_ratios/
    └── pruned_models/