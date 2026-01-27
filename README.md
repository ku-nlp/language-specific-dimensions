# Language-Specific Dimensions

Code for the paper: [arXiv:2510.07213](https://arxiv.org/abs/2510.07213)

This repo provides:
- extraction of per-language average hidden representations on FLORES-200 dev,
- representation-level intervention for multilingual generation control (English prompt → target-language output),
- evaluation notebook for layer/strength grid search,
- a playground notebook for qualitative case studies.

---
## Environment

The code is tested with the following key dependencies:

- Python == 3.12.9
- PyTorch == 2.6.0+cu124
- transformers == 4.51.3
- sacrebleu == 2.5.1
- datasets == 3.5.0
- fastText
  
  Used for language identification in evaluation.  
  Repository:
  https://github.com/facebookresearch/fastText

- repeng
  
  Used for representation-level interventions in Transformer models.  
  Repository:
  https://github.com/vgel/repeng
---
## Step 1: Extract average representations (FLORES-200 dev)

`extract.py` records the average internal representations of the model for each language and each layer. Sentences are from Flore200 dev split.

Example:
```bash
python extract.py \
  --model_key llama2-7b \
  --num_samples 50 \
  --output_path outputs/flores_hidden_stats_llama2_7b_50.json
```
---
## Step 2: Main experiments (English → target language)

`control_en2x.py` runs the main experiments in the paper.

Given the extracted statistics from **Step 1**, it:
- selects the **top-K dimensions** with the largest representation differences (language-specific dimensions),
- intervenes on these dimensions at a chosen layer with a **scaling coefficient**,
- generates translations / target-language outputs from **English inputs**.

You may need to try different **intervention layers** and **strengths** to obtain the best performance.

### Parallel setting
```bash
python control_en2x.py \
  --model_key llama2-7b \
  --setting para \
  --k_dims 400 \
  --stats_json flores_hidden_stats_llama2_7b_50.json \
  --output_root outputs \
  --target_langs zh,ja \
  --task mt
```
### Monolingual setting
```bash
python control_en2x.py \
  --model_key llama2-7b \
  --setting mono \
  --k_dims 400 \
  --stats_json flores_hidden_stats_llama2_7b_50.json \
  --output_root outputs \
  --anchor_layer 19 \
  --target_langs zh,ja,fr,de,es,ko,tr,id,ar \
  --task mt
```
Note: anchor_layer is only effective in the monolingual (mono) setting.

---
## Step 3: Evaluate grid search results

Open `compute_bleu.ipynb` and run all cells to:
- detect target-language outputs using **fastText language identification**,
- compute **BLEU** on successful samples only,
- aggregate results over **(intervention layer, scaling coefficient)** and report  
  **ACC / BLEU (success-only) / ACC × BLEU**.

---

## Playground (case studies)

Open `playground.ipynb` to interactively test different configurations, including:
- models, prompts, and settings (**para / mono**),
- target languages and **top-K dimensions**,
- intervention layers and **scaling coefficients**.

You can try your own prompt here.
