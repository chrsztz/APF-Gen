import argparse
import copy
import json
import os
import subprocess
import tempfile
from typing import Dict, Any, List

import yaml


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def save_override(base_cfg: Dict[str, Any], overrides: Dict[str, Any], path: str):
    cfg = copy.deepcopy(base_cfg)
    # shallow override for nested keys
    for k, v in overrides.items():
        sect, key = k.split(".")
        cfg.setdefault(sect, {})
        cfg[sect][key] = v
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--results", default="outputs/exp_results.json")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    experiments = {
        # Ablations
        "full": {},
        "no_physical": {"model.phys_lambda": 0.0, "data.feature_type": "base"},
        "no_attention": {"model.attention": False},
        "no_forward_planning": {"decoder.use_beam_eval": False, "inference.top_k": 1},
        "basic_model": {"data.feature_type": "base", "model.attention": False, "model.phys_lambda": 0.0},
        # Feature ablation
        "feat_all_physical": {"data.feature_type": "physical", "data.use_spatial": True, "data.use_temporal": True, "data.use_hand": True, "data.use_fingering": True},
        "feat_only_basic": {"data.feature_type": "base"},
        "feat_word2vec": {"data.feature_type": "word2vec"},
        "feat_basic_spatial": {"data.feature_type": "physical", "data.use_spatial": True, "data.use_temporal": False, "data.use_hand": False, "data.use_fingering": False},
        "feat_basic_temporal": {"data.feature_type": "physical", "data.use_spatial": False, "data.use_temporal": True, "data.use_hand": False, "data.use_fingering": False},
        "feat_basic_hand": {"data.feature_type": "physical", "data.use_spatial": False, "data.use_temporal": False, "data.use_hand": True, "data.use_fingering": False},
        "feat_basic_fingering": {"data.feature_type": "physical", "data.use_spatial": False, "data.use_temporal": False, "data.use_hand": False, "data.use_fingering": True},
        # Other models
        "cnn_base": {"data.feature_type": "base", "model.arch": "cnn_bilstm", "model.attention": False},
        "bilstm_base": {"data.feature_type": "base", "model.attention": False},
        "hybrid_base": {"data.feature_type": "base"},
        "cnn_w2v": {"data.feature_type": "word2vec", "model.attention": False},
        "bilstm_w2v": {"data.feature_type": "word2vec", "model.attention": False},
        "hybrid_w2v": {"data.feature_type": "word2vec"},
        "cnn_phys": {"data.feature_type": "physical", "model.attention": False},
        "bilstm_phys": {"data.feature_type": "physical", "model.attention": False},
        "hybrid_phys": {"data.feature_type": "physical"},
        "arlstm_phys": {"data.feature_type": "physical", "model.arch": "arlstm"},
        "argnn_phys": {"data.feature_type": "physical", "model.arch": "argnn"},
    }

    results = {}
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for name, overrides in experiments.items():
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tf:
            cfg_path = tf.name
        save_override(base_cfg, overrides, cfg_path)
        ckpt_path = os.path.join(args.checkpoint_dir, f"{name}.pt")

        # Train
        run_cmd(["python", "-m", "src.train", "--config", cfg_path])
        # Move best checkpoint to named file
        if os.path.exists("outputs/checkpoints/best.pt"):
            os.replace("outputs/checkpoints/best.pt", ckpt_path)

        # Eval
        eval_out = subprocess.check_output(
            ["python", "-m", "src.eval", "--config", cfg_path, "--checkpoint", ckpt_path],
            text=True,
        )
        results[name] = eval_out
        os.remove(cfg_path)

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()

