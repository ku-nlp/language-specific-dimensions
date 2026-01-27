import argparse
import gc
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# python extract.py --model_key llama2-7b --num_samples 50 --output_path outputs/flores_hidden_stats_llama2_7b_50.json


SUPPORTED_MODELS = {
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "aya23-8b": "CohereLabs/aya-23-8B",
}

DEFAULT_OUTPUT_PREFIX = {
    "llama2-7b": "flores_hidden_stats_llama2_7b",
    "llama2-13b": "flores_hidden_stats_llama2_13b",
    "llama3.1-8b": "flores_hidden_stats_llama3.1_8b",
    "aya23-8b": "flores_hidden_stats_aya23_8b",
}

FLORES_LANGS = {
    "en": "eng_Latn",   # English
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ja": "jpn_Jpan",   # Japanese
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "ko": "kor_Hang",   # Korean
    "tr": "tur_Latn",   # Turkish
    "id": "ind_Latn",   # Indonesian
    "ar": "arb_Arab",   # Arabic (Modern Standard)
}


@dataclass(frozen=True)
class RunningStats:
    count: int
    mean: Optional[torch.Tensor]
    m2: Optional[torch.Tensor]


def welford_update(state: RunningStats, x: torch.Tensor) -> RunningStats:
    """Online update for mean and M2 (variance accumulator)."""
    if state.count == 0:
        mean = torch.zeros_like(x)
        m2 = torch.zeros_like(x)
    else:
        mean = state.mean
        m2 = state.m2

    count = state.count + 1
    delta = x - mean
    mean = mean + delta / count
    delta2 = x - mean
    m2 = m2 + delta * delta2
    return RunningStats(count=count, mean=mean, m2=m2)


def finalize_stats(state: RunningStats) -> Dict[str, torch.Tensor]:
    if state.count < 2:
        var = torch.zeros_like(state.mean)
    else:
        var = state.m2 / (state.count - 1)
    std = torch.sqrt(var + 1e-6)
    return {"mean": state.mean, "std": std}


def resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def build_default_output_path(model_key: str, num_samples: int) -> Path:
    prefix = DEFAULT_OUTPUT_PREFIX[model_key]
    return Path(f"{prefix}_{num_samples}.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLORES inference and dump per-language per-layer hidden-state statistics."
    )
    parser.add_argument(
        "--model_key",
        type=str,
        choices=sorted(SUPPORTED_MODELS.keys()),
        default="llama2-7b",
        help="Choose one of the supported models.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of examples sampled from FLORES dev split (shared indices across languages).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "devtest", "test"])
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional explicit output path. If omitted, a default name is used.",
    )
    return parser.parse_args()


class PreNormHook:
    """Capture the final pre-norm hidden states (mean over tokens)."""

    def __init__(self) -> None:
        self.value: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.value = None

    def __call__(self, _module, inputs, _output) -> None:
        hidden = inputs[0]  # [B, T, H]
        self.value = hidden.mean(dim=1).squeeze(0).detach()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name_or_path = SUPPORTED_MODELS[args.model_key]
    output_path = args.output_path or build_default_output_path(args.model_key, args.num_samples)

    print(f"Model: {model_name_or_path}")
    print(f"Device: {args.device} | DType: {args.dtype}")
    print(f"Output: {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto" if args.device.startswith("cuda") else None,
        torch_dtype=resolve_dtype(args.dtype),
    )
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_layers = len(model.model.layers)
    last_layer_id = num_layers - 1

    # We store layer_0 ... layer_{last-1} from outputs.hidden_states,
    # and store layer_{last} as the final pre-norm state (via hook) to match your original intention.
    stats: Dict[str, Dict[str, RunningStats]] = {
        lang: {f"layer_{i}": RunningStats(count=0, mean=None, m2=None) for i in range(num_layers)}
        for lang in FLORES_LANGS.keys()
    }

    hook = PreNormHook()
    if args.model_key == "aya23-8b":
        handle = model.model.norm.register_forward_hook(hook)
    else:
        handle = model.model.layers[last_layer_id].post_attention_layernorm.register_forward_hook(hook)

    try:
        # Sample shared indices once (use English split length as reference).
        en_code = FLORES_LANGS["en"]
        en_ds = load_dataset("facebook/flores", en_code, split=args.split)
        if args.num_samples > len(en_ds):
            raise ValueError(f"--num_samples ({args.num_samples}) > dataset size ({len(en_ds)}) for split={args.split}")
        sampled_indices = random.sample(range(len(en_ds)), args.num_samples)

        for lang, flores_code in FLORES_LANGS.items():
            ds = load_dataset("facebook/flores", flores_code, split=args.split).select(sampled_indices)
            pbar = tqdm(ds, desc=f"FLORES {lang} ({args.split})", leave=False)

            for ex in pbar:
                text = ex["sentence"]
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                inputs = {k: v.to(args.device) for k, v in inputs.items()}

                hook.clear()
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states  # tuple of [B, T, H]

                # Update layers 0..last-1 from hidden_states (mean over tokens).
                for layer_id in range(last_layer_id):
                    vec = hidden_states[layer_id].mean(dim=1).squeeze(0).detach().cpu().float()
                    key = f"layer_{layer_id}"
                    stats[lang][key] = welford_update(stats[lang][key], vec)

                # Update layer_{last} with final pre-norm state (hook).
                if hook.value is None:
                    hidden_dim = hidden_states[0].shape[-1]
                    pre_norm_vec = torch.zeros(hidden_dim)
                else:
                    pre_norm_vec = hook.value.detach().cpu().float()

                stats[lang][f"layer_{last_layer_id}"] = welford_update(
                    stats[lang][f"layer_{last_layer_id}"], pre_norm_vec
                )

                del inputs, outputs, hidden_states
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

    finally:
        handle.remove()

    output = {"summary": {}}
    for lang in FLORES_LANGS.keys():
        output["summary"][lang] = {}
        for layer_key, state in stats[lang].items():
            final = finalize_stats(state)
            output["summary"][lang][layer_key] = {
                "mean": final["mean"].tolist(),
                "std": final["std"].tolist(),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Saved to: {output_path}")


if __name__ == "__main__":
    main()
