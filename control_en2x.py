import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from repeng import ControlVector, ControlModel
from repeng.control import ControlModule

# Para
# python control_en2x.py \
#   --model_key llama2-7b \
#   --setting para \
#   --k_dims 400 \
#   --stats_json flores_hidden_stats_llama2_7b_50.json \
#   --output_root outputs \
#   --target_langs zh,ja \
#   --task mt

# Mono (anchor default = 19)
# python control_en2x.py \
#   --model_key llama2-7b \
#   --setting mono \
#   --k_dims 400 \
#   --stats_json flores_hidden_stats_llama2_7b_50.json \
#   --output_root outputs
#   --anchor_layer 19 \
#   --target_langs zh,ja,fr,de,es,ko,tr,id,ar \
#   --task mt


DEFAULT_COEFF_MAP: Dict[int, np.ndarray] = {
    18: [0.4]
    # 18: np.linspace(0.2, 0.8, 4),
    # 20: np.linspace(0.4, 1.2, 5),
    # 22: np.linspace(0.6, 1.4, 5),
    # 24: np.linspace(0.6, 1.8, 7),
    # 26: np.linspace(0.8, 2.0, 7),
    # 28: np.linspace(1.0, 2.2, 7),
    # 30: np.linspace(1.0, 2.2, 7),
}

SUPPORTED_MODELS: Dict[str, str] = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "aya23-8b": "CohereLabs/aya-23-8B",
}


LANG_NAME: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ko": "Korean",
    "tr": "Turkish",
    "id": "Indonesian",
    "ar": "Arabic",
}


@dataclass(frozen=True)
class Config:
    model_key: str
    setting: str  # "para" or "mono"
    k_dims: int
    stats_json: Path
    data_dir: Path
    output_root: Path
    wrap_start_layer: int
    anchor_layer: int
    max_new_tokens: int
    dtype: str
    device: str
    target_langs: List[str]
    task: str  # "mt" or "mlqa"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="En-to-X control via language-specific dimensions (repeng)."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        choices=sorted(SUPPORTED_MODELS.keys()),
        required=True,
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["para", "mono"],
        default="para",
    )
    parser.add_argument("--k_dims", type=int, default=400)
    parser.add_argument("--stats_json", type=Path, required=True)

    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs"))

    parser.add_argument(
        "--wrap_start_layer",
        type=int,
        default=6,
        help="Wrap layers [wrap_start_layer, num_layers) with ControlModel.",
    )
    parser.add_argument(
        "--anchor_layer",
        type=int,
        default=19,
        help="Only used in mono setting: the anchor layer treated as the English proxy.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--target_langs",
        type=str,
        default="ja,zh,es,fr,de",
        help="Comma-separated target langs (e.g., ja,zh,es,fr,de).",
    )
    parser.add_argument("--task", type=str, choices=["mt", "mlqa"], default="mt")

    args = parser.parse_args()
    target_langs = [x.strip() for x in args.target_langs.split(",") if x.strip()]

    return Config(
        model_key=args.model_key,
        setting=args.setting,
        k_dims=args.k_dims,
        stats_json=args.stats_json,
        data_dir=args.data_dir,
        output_root=args.output_root,
        wrap_start_layer=args.wrap_start_layer,
        anchor_layer=args.anchor_layer,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        target_langs=target_langs,
        task=args.task
    )


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_summary(stats_json: Path) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    with stats_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["summary"]


def load_dataset_mlqa(lang: str, data_dir: Path) -> Tuple[List[dict], List[str]]:
    path = data_dir / "mlqa" / "en.json"
    recs: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    target_name = LANG_NAME.get(lang, lang)

    prompts = [
        (
            "Answer the question using a short phrase directly copied from the context, "
            "translated into the target language.\n\n"
            "Do not explain or generate additional text.\n\n"
            f"Context:\n{r['context']}\n\n"
            f"Question:\n{r['question']}\n\n"
            f"Target language:\n{target_name}\n"
        )
        for r in recs
    ]
    return recs, prompts


def load_dataset(lang: str, data_dir: Path) -> Tuple[List[dict], List[str]]:
    path = data_dir / "trans" / f"en-{lang}.pkl"
    with path.open("rb") as f:
        records = pickle.load(f)  # list[dict], must contain: 'en', lang, 'source'

    lang_name = LANG_NAME.get(lang, lang)
    prompts = [
        f"Translate an English sentence into {lang_name}.\n"
        f"English: {r['en']}\n"
        f"{lang_name}: "
        for r in records
    ]
    return records, prompts


def load_task_dataset(task: str, lang: str, data_dir: Path) -> Tuple[List[dict], List[str]]:
    if task == "mlqa":
        return load_dataset_mlqa(lang, data_dir)
    return load_dataset(lang, data_dir)


def apply_model_specific_zeroing(model_key: str, diff: np.ndarray) -> None:
    if model_key == "llama2-7b":
        if diff.shape[0] > 1415:
            diff[1415] = 0.0
        if diff.shape[0] > 2533:
            diff[2533] = 0.0
    elif model_key == "llama2-13b":
        if diff.shape[0] > 2100:
            diff[2100] = 0.0


def build_control_vector_for_lang(
    summary: Dict[str, Dict[str, Dict[str, List[float]]]],
    model_key: str,
    setting: str,
    lang: str,
    k_dims: int,
    last_layer_id: int,
    anchor_layer: int,
) -> np.ndarray:
    """
    Build En-to-X control vector for one language.

    - para: source = en@last, target = lang@last
    - mono: source = lang@anchor (English proxy), target = lang@last

    Select top-K by |source - target|, then set control vector to target_mean on those dims.
    """
    target_key = f"layer_{last_layer_id}"
    target_mean = np.asarray(summary[lang][target_key]["mean"], dtype=np.float32)

    if setting == "mono":
        source_key = f"layer_{anchor_layer}"
        source_mean = np.asarray(summary[lang][source_key]["mean"], dtype=np.float32)
    else:
        source_mean = np.asarray(summary["en"][target_key]["mean"], dtype=np.float32)

    diff = source_mean - target_mean
    apply_model_specific_zeroing(model_key, diff)

    topk_idx = np.argsort(-np.abs(diff))[:k_dims]
    control_vec = np.zeros_like(target_mean, dtype=np.float32)
    control_vec[topk_idx] = target_mean[topk_idx]
    return control_vec


def build_output_folder(
    output_root: Path,
    model_key: str,
    setting: str,
    k_dims: int,
    anchor_layer: int,
) -> Path:
    if setting == "mono":
        return output_root / f"{model_key}_dims{k_dims}_{setting}_anchorL{anchor_layer}"
    return output_root / f"{model_key}_dims{k_dims}_{setting}"


def patch_control_forward() -> None:
    """
    Overwrite ControlModule.forward with overwrite-nonzero behavior.
    ControlModule.custom_coeff scales the injected control vector.
    """
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
        control = control.to(modified.device)

        norm_pre = torch.norm(modified, dim=-1, keepdim=True)

        if "position_ids" in kwargs:
            pos = kwargs["position_ids"]
            zero_idx = (pos == 0).cumsum(1).argmax(1, keepdim=True)
            col = torch.arange(pos.size(1), device=pos.device).unsqueeze(0)
            mask = (col >= zero_idx).float().reshape(modified.shape[0], modified.shape[1], 1)
            mask = mask.to(modified.dtype).to(modified.device)
        else:
            mask = 1.0

        control_applied = control * float(ControlModule.custom_coeff) * mask
        modified = torch.where(control_applied != 0, control_applied, modified)

        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    ControlModule.forward = normalized_forward


@torch.no_grad()
def main() -> None:
    cfg = parse_args()

    model_name_or_path = SUPPORTED_MODELS[cfg.model_key]
    print(f"Model: {model_name_or_path}")
    print(f"Setting: {cfg.setting} | K: {cfg.k_dims}")
    if cfg.setting == "mono":
        print(f"Anchor layer: {cfg.anchor_layer}")
    print(f"Stats: {cfg.stats_json}")
    print(f"Device: {cfg.device} | DType: {cfg.dtype}")

    summary = load_summary(cfg.stats_json)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto" if cfg.device.startswith("cuda") else None,
        torch_dtype=resolve_dtype(cfg.dtype),
    )
    base_model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_layers = len(base_model.model.layers)
    last_layer_id = num_layers - 1

    patch_control_forward()

    wrap_start = min(max(cfg.wrap_start_layer, 0), num_layers)
    wrapped_layers = list(range(wrap_start, num_layers))
    model = ControlModel(base_model, wrapped_layers)

    gen_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": cfg.max_new_tokens,
    }

    def truncate_at_stop(s: str) -> str:
        idx = s.find("\n")
        return s if idx == -1 else s[:idx]

    output_folder = build_output_folder(
        cfg.output_root,
        cfg.model_key,
        cfg.setting,
        cfg.k_dims,
        cfg.anchor_layer,
    )
    output_folder = output_folder / cfg.task
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Outputs: {output_folder}")

    # Precompute En-to-X control vectors for this setting.
    control_vec_by_lang: Dict[str, np.ndarray] = {}
    for lang in cfg.target_langs:
        if lang == "en":
            continue
        control_vec_by_lang[lang] = build_control_vector_for_lang(
            summary=summary,
            model_key=cfg.model_key,
            setting=cfg.setting,
            lang=lang,
            k_dims=cfg.k_dims,
            last_layer_id=last_layer_id,
            anchor_layer=cfg.anchor_layer,
        )

    # Iterate only valid layers; skip invalid safely.
    for layer_idx in range(wrap_start, num_layers):
        if layer_idx not in DEFAULT_COEFF_MAP:
            continue

        coeff_list = DEFAULT_COEFF_MAP[layer_idx]
        print(f"Layer {layer_idx}: coeffs = {list(map(float, coeff_list))}")

        for coeff in coeff_list:
            ControlModule.custom_coeff = float(coeff)

            for lang in cfg.target_langs:
                if lang == "en":
                    continue
                if lang not in control_vec_by_lang:
                    continue

                # Control is applied only at layer_idx; other layers receive zeros.
                layer_directions: Dict[int, np.ndarray] = {}
                for lid in range(num_layers):
                    layer_directions[lid] = control_vec_by_lang[lang] if lid == layer_idx else np.zeros_like(
                        control_vec_by_lang[lang]
                    )

                control_vector = ControlVector(model_type="llama", directions=layer_directions)

                model.reset()
                model.set_control(control_vector, coeff=1.0)

                records, prompts = load_task_dataset(cfg.task, lang, cfg.data_dir)

                results: List[dict] = []
                print(f"Running {lang} | layer={layer_idx} | coeff={coeff:.2f}")

                for idx, prompt in enumerate(tqdm(prompts, desc=f"{lang} prompts", leave=False)):
                    rec = records[idx]

                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    out = model.generate(**inputs, **gen_kwargs)
                    gen_tokens = out[0][inputs["input_ids"].shape[-1] :]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    text = truncate_at_stop(text).strip()

                    # results.append(
                    #     {
                    #         "source": rec["en"],
                    #         "refer": rec[lang],
                    #         "trans": text,
                    #         "data": rec["source"],
                    #     }
                    # )
                    if cfg.task == "mlqa":
                        gold = None
                        if "answers" in rec and isinstance(rec["answers"], dict) and "text" in rec["answers"]:
                            gold = rec["answers"]["text"]
                        elif "answer" in rec:
                            gold = rec["answer"]
                        elif "gold" in rec:
                            gold = rec["gold"]

                        results.append(
                            {
                                "task": "mlqa",
                                "lang": lang,
                                "id": rec.get("id"),
                                "question": rec.get("question"),
                                "context": rec.get("context"),
                                "gold": gold,
                                "pred": text,
                            }
                        )
                    else:
                        results.append(
                            {
                                "task": "mt",
                                "source": rec["en"],
                                "refer": rec[lang],
                                "trans": text,
                                "data": rec["source"],
                            }
                        )

                    if cfg.device.startswith("cuda"):
                        torch.cuda.empty_cache()

                # Keep the naming scheme, but save as JSON for easier viewing.
                out_path = os.path.join(
                    str(output_folder),
                    f"en.control.{layer_idx}.{float(coeff):.2f}.{lang}.json",
                )
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"Saved: {out_path}")

    # If DEFAULT_COEFF_MAP contains layers beyond num_layers, they are ignored by design.


if __name__ == "__main__":
    main()