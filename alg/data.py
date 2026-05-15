import json

import torch
from torch.utils.data import Dataset

from config import ACGPFConfig


class MultiTeacherCoTDataset(Dataset):
    """JSON: list of {Question, CoT: [str,...], Answer}."""

    def __init__(self, json_path, tokenizer, max_len=None):
        max_len = max_len or ACGPFConfig.max_seq_len
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.data = list(raw_data)
        self.tokenizer = tokenizer
        self.max_len = max_len

        thinking_ids_list = []
        for word in ACGPFConfig.thinking_keywords:
            ids = tokenizer([word, " " + word], add_special_tokens=False).input_ids
            for id_list in ids:
                if len(id_list) == 1:
                    thinking_ids_list.append(id_list[0])
        self.thinking_ids = torch.tensor(list(set(thinking_ids_list)), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["Question"]
        cots = item["CoT"]
        gold_answer = item["Answer"]

        processed_inputs = []
        for cot_text in cots:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": cot_text},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            processed_inputs.append(text)

        return {"inputs": processed_inputs, "gold_answer": gold_answer}


def acgpf_collate_fn(batch, tokenizer):
    assert len(batch) == 1, "Batch size must be 1 for this collate."

    item = batch[0]
    texts = item["inputs"]
    gold_answer = item["gold_answer"]

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=ACGPFConfig.max_seq_len,
    )

    assistant_marker = "<|im_start|>assistant\n"
    marker_ids = tokenizer(assistant_marker, add_special_tokens=False).input_ids
    marker_len = len(marker_ids)

    assistant_starts = []
    for i in range(encodings.input_ids.size(0)):
        input_ids = encodings.input_ids[i].tolist()
        found_pos = -1
        for pos in range(len(input_ids) - marker_len + 1):
            if input_ids[pos : pos + marker_len] == marker_ids:
                found_pos = pos + marker_len
                break
        assistant_starts.append(found_pos if found_pos >= 0 else 0)

    gold_enc = tokenizer(gold_answer, return_tensors="pt", add_special_tokens=False)

    return {
        "input_ids": encodings.input_ids,
        "attention_mask": encodings.attention_mask,
        "gold_input_ids": gold_enc.input_ids,
        "assistant_start_positions": torch.tensor(assistant_starts, dtype=torch.long),
    }
