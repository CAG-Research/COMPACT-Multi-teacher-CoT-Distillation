# COMPACT `alg`: Multi-Teacher CoT Distillation Training

This directory contains training scripts built on **Accelerate + DeepSpeed + PEFT LoRA**, implementing weighted supervision and a consistency-style regularizer over multiple teacher Chain-of-Thought trajectories per question (trainer class `ACGPFTrainer`, config `ACGPFConfig`).

---

## Is the project “complete”?

### What is implemented

| Module | Role |
|--------|------|
| `config.py` | Hyperparameters and env overrides; LoRA; epochs; loss weights |
| `data.py` | Load JSON dataset, build multi-teacher `CoT` chat strings; collect “thinking” token IDs |
| `trainer.py` | Forward pass, per-teacher cross-entropy, MI-style scores, consensus scores, teacher weights `alphas`, optional MCC (symmetric KL) |
| `train.py` | Model/data loading, distributed sampler, training loop, epoch checkpoints |
| `checkpoint.py` | Gather LoRA state under DeepSpeed ZeRO and `save_pretrained` |
| `mt_cot_kd.py` | Entry point: calls `train.main()` |

For a minimal **train end-to-end** pipeline, the loop (data → forward → backward → save) is in place.

### Gaps and caveats (read before use)

1. **No dependency manifest**  
   There is no `requirements.txt` / `pyproject.toml`; install packages matching the imports (see below).

2. **Defaults point to another machine**  
   `ACGPFConfig` defaults for `model_name` and `dataset_path` use `/data/gjq/...`. On a new host, set environment variables or edit the config.

3. **`output_path` is unused**  
   `config.output_path` exists, but `train.py` only uses `checkpoint_dir`. Add logic if you want outputs separate from checkpoints.

4. **Tight coupling to model layout and chat template**  
   - `trainer.py` assumes a Llama/Qwen-like stack: `model.layers[-1].self_attn.q_proj/k_proj`.  
   - `data.py` hard-codes the assistant marker prefix `"<|im_start|>assistant\n"` (Qwen-style chat). Change it when switching models/tokenizers.

5. **Effective `batch_size` is 1**  
   `acgpf_collate_fn` asserts `len(batch) == 1`. Multiple teachers are stacked along the **first dimension** for one question–batch item. With `DistributedSampler` / multi-GPU, each rank still sees one such item per step—not classic `batch_size > 1` sample parallelism across questions.

6. **No evaluation or inference scripts**  
   Training only; add scripts if you need dev-set metrics or a bundled inference API.

7. **`flash_attention_2`**  
   `train.py` sets `attn_implementation="flash_attention_2"`. Builds without FA2 support will fail; switch to `sdpa` or omit the argument per your `transformers` version.

---

## Data format

The JSON file must be a **list** of objects like:

```json
{
  "Question": "Question text",
  "CoT": ["Teacher 1 reasoning…", "Teacher 2 reasoning…"],
  "Answer": "Gold answer (used for gold tokenization and parts of the loss)"
}
```

Field names must match `data.py`: `Question`, `CoT`, `Answer`.

---

## Environment variables (optional)

| Variable | Meaning | Default in code |
|----------|---------|------------------|
| `MODEL_NAME` | HF model id or local path | `/data/gjq/.../qwen2_5-1_5b` |
| `DATASET_PATH` | Path to the JSON above | `/data/gjq/.../sampled_200.json` |
| `OUTPUT_PATH` | Not used by `train.py` today | `./output/qwen2_5-1_5b` |
| `CHECKPOINT_DIR` | Checkpoint root | `./checkpoints` |
| `MAX_SEQ_LEN` | Max sequence length | `2560` |
| `LEARNING_RATE` | AdamW LR | `1e-5` |
| `BETA_1` / `BETA_2` / `BETA_3` | MI / consensus / perplexity weights | `1.0` / `1.0` / `0.5` |
| `TEMPERATURE` | Softmax temperature for teacher weights | `1.0` |
| `LAMBDA_MCC` | MCC symmetric-KL coefficient | `0.1` |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | LoRA hyperparameters | `16` / `32` / `0.1` |
| `BATCH_SIZE` | DataLoader batch (must be 1) | `1` |
| `GRAD_ACCUM_STEPS` | Gradient accumulation steps | `4` |
| `EPOCHS` | Training epochs | `5` |
| `SAVE_EPOCHS` | Save every N epochs | `1` |

---

## Dependencies (inferred)

Direct imports:

- `torch`
- `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`, optional FA2)
- `accelerate` (including `DeepSpeedPlugin`)
- `peft` (`LoraConfig`, `get_peft_model`)
- `deepspeed` (ZeRO gather in checkpoint save)

Example install (pick versions for your CUDA stack):

```bash
pip install torch transformers accelerate peft deepspeed
```

If Flash Attention 2 is unavailable, edit `attn_implementation` in `train.py`.

---

## How to run

Assuming CUDA and (optionally) a multi-process launcher are configured:

```bash
cd /data/cj/COMPACT/alg
export MODEL_NAME=/path/to/your/model
export DATASET_PATH=/path/to/your_dataset.json

# Single-GPU example (if you omit DeepSpeed, adjust Accelerator setup accordingly)
python train.py

# Equivalent to mt_cot_kd.py
python mt_cot_kd.py
```

Multi-GPU is typically `accelerate launch` or `torchrun` with a matching DeepSpeed config; no `accelerate` config ships in this folder—supply one per your cluster policy.

---

## File layout

```
alg/
├── README.md          # This document
├── config.py          # ACGPFConfig
├── data.py            # MultiTeacherCoTDataset + collate
├── trainer.py         # ACGPFTrainer (losses and weights)
├── train.py           # Main training script
├── checkpoint.py      # LoRA + ZeRO-3 saves
└── mt_cot_kd.py       # Thin entry wrapper
```

---

## Summary

As a prototype, **`alg`** is reasonably complete for **training and checkpointing**. For a turnkey experiment repo, consider adding **locked dependencies**, **sane default paths or a copy-paste env example**, **evaluation or a smoke test**, and verifying **chat markers** and **`forward` backbone assumptions** for your target model.
