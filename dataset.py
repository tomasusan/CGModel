

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        lang2id: Optional[Dict[str, int]] = None,
        code_field: str = "code",
        prompt_field: str = "prompt",
        response_field: str = "response",
    ):
        """
        Args:
            data_path: .parquet or .json file path containing records.
            tokenizer: huggingface tokenizer
            max_seq_len: maximum sequence length
            lang2id: optional mapping from language string to int id
            code_field: which field in the source contains pure code used for AST
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lang2id = lang2id or {"python": 0, "java": 1, "cpp": 2, "js": 3}
        self.code_field = code_field
        self.prompt_field = prompt_field
        self.response_field = response_field

        # load
        if data_path.endswith(".parquet"):
            self._load_parquet()
        elif data_path.endswith(".json"):
            self._load_json()
        else:
            raise ValueError(f"Unsupported file: {data_path}")

        # filter samples where prompt alone is too long
        self._prefilter()

        print(f"Loaded dataset: {len(self.data)} samples from {data_path}")

    def _load_parquet(self):
        df = pd.read_parquet(self.data_path)
        items = []
        for _, row in df.iterrows():
            items.append({
                "prompt": str(row.get(self.prompt_field, "")),
                "response": str(row.get(self.response_field, "")),
                "code": str(row.get(self.code_field, row.get(self.response_field, ""))),
                "language": str(row.get("programming_language", row.get("language", "python"))).lower(),
                "difficulty": str(row.get("adjective", row.get("difficulty", "unknown")))
            })
        self.data = items

    def _load_json(self):
        import json
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        for item in raw:
            # flexible parsing
            if self.prompt_field in item and self.response_field in item:
                prompt = item[self.prompt_field]
                response = item[self.response_field]
                code = item.get(self.code_field, response)
                lang = item.get("programming_language", item.get("language", "python"))
            elif "instruction" in item and "output" in item:
                # older format
                prompt = item.get("instruction", "")
                if item.get("input"):
                    prompt = prompt + "\n" + item.get("input")
                response = item.get("output", "")
                code = item.get(self.code_field, response)
                lang = item.get("language", "python")
            else:
                # skip unknown formats
                continue

            items.append({
                "prompt": str(prompt),
                "response": str(response),
                "code": str(code),
                "language": str(lang).lower(),
                "difficulty": str(item.get("difficulty", "unknown"))
            })

        self.data = items

    def _prefilter(self):
        """Filter out samples where the prompt tokens alone exceed max_seq_len.
        This prevents generating labels that are all -100 or accidentally training on prompt.
        """
        kept = []
        for it in self.data:
            prompt_enc = self.tokenizer(it["prompt"], add_special_tokens=False)
            if len(prompt_enc["input_ids"]) < self.max_seq_len:
                kept.append(it)
        self.data = kept

    def __len__(self):
        return len(self.data)

    def format_prompt(self, prompt: str) -> str:
        # Keep template simple; avoid adding special tokens here; let tokenizer handle special tokens
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def format_conversation(self, prompt: str, response: str) -> str:
        return self.format_prompt(prompt) + response + (self.tokenizer.eos_token or "")

    def __getitem__(self, idx):
        item = self.data[idx]

        # build conversation text
        conversation = self.format_conversation(item["prompt"], item["response"])

        tokens = self.tokenizer(
            conversation,
            add_special_tokens=False,
            max_length=self.max_seq_len,
            truncation=True,
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # compute prompt length for masking
        prompt_text = self.format_prompt(item["prompt"])
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        labels = torch.tensor(input_ids)
        labels[:prompt_len] = -100
        labels = labels.clone()
        labels[tokens["attention_mask"] == 0] = -100

        # padding
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels = torch.cat([labels, torch.full((pad_len,), -100)])

        language = item.get("language", "python").lower()
        language_id = torch.tensor(self.lang2id.get(language, 0), dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": labels,
            "language_id": language_id,
            "code": item.get("code")
        }