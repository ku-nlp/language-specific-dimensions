import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pickle
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlModel, ControlVector
from repeng.control import ControlModule


# -----------------------------
# Config
# -----------------------------
SUPPORTED_MODELS: Dict[str, str] = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "aya23-8b": "CohereLabs/aya-23-8B",
}

LANG_PROMPTS = {
    "ja": {
        "instruction": "日本語の文を英語に翻訳してください。",
        "src_label": "日本語",
        "src_key": "ja",
        "tgt_label": "English",
    },
    "zh": {
        "instruction": "请将下面的中文句子翻译成英文。",
        "src_label": "中文",
        "src_key": "zh",
        "tgt_label": "English",
    },
    "es": {
        "instruction": "Traduce la siguiente frase en español al inglés.",
        "src_label": "Español",
        "src_key": "es",
        "tgt_label": "English",
    },
    "fr": {
        "instruction": "Traduisez la phrase française suivante en anglais.",
        "src_label": "Français",
        "src_key": "fr",
        "tgt_label": "English",
    },
    "de": {
        "instruction": "Übersetzen Sie den folgenden deutschen Satz ins Englische.",
        "src_label": "Deutsch",
        "src_key": "de",
        "tgt_label": "English",
    },
    "ko": {
        "instruction": "다음 한국어 문장을 영어로 번역하세요.",
        "src_label": "한국어",
        "src_key": "ko",
        "tgt_label": "English",
    },
    "tr": {
        "instruction": "Aşağıdaki Türkçe cümleyi İngilizceye çevirin.",
        "src_label": "Türkçe",
        "src_key": "tr",
        "tgt_label": "English",
    },
    "id": {
        "instruction": "Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Inggris.",
        "src_label": "Bahasa Indonesia",
        "src_key": "id",
        "tgt_label": "English",
    },
    "ar": {
        "instruction": "ترجم الجملة العربية التالية إلى الإنجليزية.",
        "src_label": "العربية",
        "src_key": "ar",
        "tgt_label": "English",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", type=str, choices=SUPPORTED_MODELS.keys(), required=True)
    p.add_argument("--stats_json", type=str, required=True)
    p.add_argument("--k_dims", type=int, default=400)
    p.add_argument("--target_langs", type=str, default="ja,zh,es,fr,de")
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--task", type=str, default="mt")  # reserved (future: mlqa, etc.)
    return p.parse_args()


# -----------------------------
# ControlModule patch (keep your behavior)
# -----------------------------
original_forward = ControlModule.forward
ControlModule.custom_coeff = 1.0


def normalized_forward(self, *args, **kwargs):
    output = self.block(*args, **kwargs)
    control = self.params.control

    if control is None:
        return output
    if len(control.shape) == 1:
        control = control.reshape(1, 1, -1)

    if torch.all(control == 0):
        return output

    modified = output[0] if isinstance(output, tuple) else output

    assert len(control.shape) == len(modified.shape)
    control = control.to(modified.device)

    norm_pre = torch.norm(modified, dim=-1, keepdim=True)

    if "position_ids" in kwargs:
        pos = kwargs["position_ids"]
        zero_indices = (pos == 0).cumsum(1).argmax(1, keepdim=True)
        col_indices = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
        mask = (col_indices >= zero_indices).float().reshape(modified.shape[0], modified.shape[1], 1)
        mask = mask.to(modified.dtype).to(modified.device)
    else:
        mask = 1.0

    non_zero_mask = (control != 0).float()
    non_zero_count = non_zero_mask.sum()
    if non_zero_count > 0:
        control = control * self.custom_coeff

    control_applied = control * mask
    modified = torch.where(control_applied != 0, control_applied, modified)

    if self.params.normalize:
        norm_post = torch.norm(modified, dim=-1, keepdim=True)
        modified = modified / norm_post * norm_pre

    if isinstance(output, tuple):
        return (modified,) + output[1:]
    return modified


# -----------------------------
# Data loading (MT only now; task interface reserved)
# -----------------------------
def load_dataset_mt(src_lang: str, data_dir: Path) -> Tuple[List[dict], List[str]]:
    if src_lang not in LANG_PROMPTS:
        raise ValueError(f"Unsupported language: {src_lang}")

    cfg = LANG_PROMPTS[src_lang]
    # data/assets_translation/merged/en-<lang>.pkl (relative path)
    pkl_path = data_dir / "assets_translation" / "merged" / f"en-{src_lang}.pkl"

    with pkl_path.open("rb") as f:
        records = pickle.load(f)

    prompts = [
        (
            f"{cfg['instruction']}\n"
            f"{cfg['src_label']}: {r[cfg['src_key']]}\n"
            f"{cfg['tgt_label']}: "
        )
        for r in records
    ]
    return records, prompts


def load_task_dataset(task: str, src_lang: str, data_dir: Path) -> Tuple[List[dict], List[str]]:
    # reserved for future tasks; for now MT only
    if task != "mt":
        raise ValueError(f"Task '{task}' is not implemented in this script yet.")
    return load_dataset_mt(src_lang, data_dir)


# -----------------------------
# Utilities
# -----------------------------
def truncate_at_newline(text: str) -> str:
    idx = text.find("\n")
    return text if idx == -1 else text[:idx]


def build_output_dir(output_root: Path, model_key: str, k_dims: int) -> Path:
    # You can rename this folder freely; it does NOT affect per-file naming.
    # Make it explicit: direction, K, setting
    name = f"x2en_dims{k_dims}_para_{model_key}"
    out_dir = output_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_model_layer_count(hf_model) -> int:
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return len(hf_model.model.layers)
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "decoder") and hasattr(hf_model.model.decoder, "layers"):
        return len(hf_model.model.decoder.layers)
    raise ValueError("Cannot infer layer count from model.")


def get_layer_key(layer_idx: int) -> str:
    return f"layer_{layer_idx}"


def load_stats(stats_json_path: Path) -> dict:
    with stats_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def make_control_vectors_for_x_to_en(
    summary: dict,
    src_langs: List[str],
    k_dims: int,
    model_key: str,
    final_layer_idx: int,
) -> Dict[str, np.ndarray]:
    """
    Build per-language control vectors (X -> En).
    We select top-K dimensions from diff, then set those dimensions to EN mean values.
    """
    en_mean = np.array(summary["en"][get_layer_key(final_layer_idx)]["mean"], dtype=np.float32)
    control_vecs: Dict[str, np.ndarray] = {}

    for lang in src_langs:
        # diff definition aligns with your previous pipeline:
        # diff = anchor_mean - lang_final_mean
        # for X->En, we still only need top-K indices; actual injected values are EN mean.
        anchor_mean = np.array(summary["en"][get_layer_key(final_layer_idx)]["mean"], dtype=np.float32)
        lang_final = np.array(summary[lang][get_layer_key(final_layer_idx)]["mean"], dtype=np.float32)

        diff = anchor_mean - lang_final

        # model-specific masking
        if model_key == "llama2-7b":
            diff[1415] = 0.0
            diff[2533] = 0.0
        elif model_key == "llama2-13b":
            diff[2100] = 0.0

        topk = np.argsort(-np.abs(diff))[:k_dims]

        vec = np.zeros_like(en_mean, dtype=np.float32)
        vec[topk] = en_mean[topk]
        control_vecs[lang] = vec

    return control_vecs


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    model_name = SUPPORTED_MODELS[args.model_key]
    stats_path = Path(args.stats_json)
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)

    src_langs = [s.strip() for s in args.target_langs.split(",") if s.strip()]
    for l in src_langs:
        if l not in LANG_PROMPTS:
            raise ValueError(f"Language '{l}' not in LANG_PROMPTS. Please add it.")

    # Load model/tokenizer (HF auth assumed via CLI login)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.eval()

    n_layers = get_model_layer_count(base_model)
    final_layer_idx = n_layers - 1

    # Patch ControlModule.forward
    ControlModule.forward = normalized_forward

    # Wrap with ControlModel; keep your original layer range idea but clip by n_layers
    start_layer = 6
    end_layer = n_layers
    control_layers = list(range(start_layer, end_layer))
    model = ControlModel(base_model, control_layers)

    # Load stats
    summary = load_stats(stats_path)

    # Prepare control vectors (X -> En)
    control_vecs = make_control_vectors_for_x_to_en(
        summary=summary,
        src_langs=src_langs,
        k_dims=args.k_dims,
        model_key=args.model_key,
        final_layer_idx=final_layer_idx,
    )

    # Coeff map (keep your values; will be filtered by n_layers)
    coeff_map: Dict[int, np.ndarray] = {
        12: np.linspace(0.2, 1.0, 5),
        16: np.linspace(0.2, 1.0, 5),
        18: np.linspace(0.2, 1.2, 6),
        20: np.linspace(0.4, 1.4, 6),
        22: np.linspace(0.4, 1.6, 7),
        24: np.linspace(0.6, 1.8, 7),
        26: np.linspace(0.6, 1.8, 7),
        28: np.linspace(0.8, 2.0, 7),
        30: np.linspace(0.8, 2.0, 7),
    }

    out_dir = build_output_dir(output_root, args.model_key, args.k_dims)

    gen_kwargs = {
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128,
    }

    # Main sweep
    for layer_idx, coeff_list in coeff_map.items():
        if layer_idx < 0 or layer_idx >= n_layers:
            print(f"Skip layer {layer_idx}: out of range for this model (n_layers={n_layers}).")
            continue

        print(f"Processing layer {layer_idx} with coefficients: {coeff_list.tolist()}")

        for coeff in coeff_list:
            ControlModule.custom_coeff = float(coeff)

            for lang in src_langs:
                vec = control_vecs[lang]

                directions = {
                    lid: (vec if lid == layer_idx else np.zeros_like(vec))
                    for lid in range(n_layers)
                }
                control_vector = ControlVector(model_type="llama", directions=directions)

                model.reset()
                model.set_control(control_vector, coeff=1)

                records, prompts = load_task_dataset(args.task, lang, data_dir)

                results = []
                print(f"  -> Lang={lang}, coeff={coeff:.2f}")

                for idx, prompt in enumerate(tqdm(prompts, desc=f"{lang} prompts")):
                    rec = records[idx]

                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    out = model.generate(**inputs, **gen_kwargs)
                    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
                    pred = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    pred = truncate_at_newline(pred).strip()

                    # X -> En direction
                    results.append(
                        {
                            "source": rec[lang],   # source sentence in X
                            "refer": rec["en"],    # English reference
                            "trans": pred,         # model output
                            "data": rec.get("source"),
                        }
                    )

                out_path = out_dir / f"{lang}.control.{layer_idx}.{float(coeff):.2f}.en.jsonl"
                with out_path.open("w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

                print(f"[+] Saved: {out_path}")

                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()