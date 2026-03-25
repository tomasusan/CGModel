import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) Dataset for code generation tasks.

    This dataset loads code generation examples from parquet or JSON files,
    tokenizes them, and prepares them for causal language modeling training.
    It handles prompt-response formatting, language identification, and
    proper masking of prompt tokens during loss computation.
    """

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
        Initialize the SFT dataset.

        Args:
            data_path: Path to .parquet or .json file containing training records
            tokenizer: HuggingFace tokenizer instance for text tokenization
            max_seq_len: Maximum sequence length for tokenization (truncation/padding)
            lang2id: Optional mapping from language strings to integer IDs for AST processing
            code_field: Field name in source data that contains pure code for AST evaluation
            prompt_field: Field name containing the instruction/prompt
            response_field: Field name containing the expected model response
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # Default language to ID mapping if not provided
        self.lang2id = lang2id or {"python": 0, "java": 1, "cpp": 2, "js": 3}
        self.code_field = code_field
        self.prompt_field = prompt_field
        self.response_field = response_field

        # Load data based on file extension
        if data_path.endswith(".parquet"):
            self._load_parquet()
        elif data_path.endswith(".json"):
            self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Filter out samples where prompt alone exceeds max sequence length
        self._prefilter()

        print(f"Loaded dataset: {len(self.data)} samples from {data_path}")

    def _load_parquet(self):
        """
        Load and parse data from a parquet file.

        Extracts prompt, response, code, language, and difficulty fields
        from each row and stores them as a list of dictionaries.
        """
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

        print(f"Loaded {len(self.data)} samples from parquet")

    def _load_json(self):
        """
        Load and parse data from a JSON file.

        Supports both standard format (prompt/response fields) and
        legacy format (instruction/input/output fields) for flexibility.
        """
        import json
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        for item in raw:
            # Flexible parsing to support multiple data formats
            if self.prompt_field in item and self.response_field in item:
                # Standard format with explicit prompt/response fields
                prompt = item[self.prompt_field]
                response = item[self.response_field]
                code = item.get(self.code_field, response)
                lang = item.get("programming_language", item.get("language", "python"))
            elif "instruction" in item and "output" in item:
                # Legacy format with instruction/input/output
                prompt = item.get("instruction", "")
                # Append input to prompt if present (common in instruction datasets)
                if item.get("input"):
                    prompt = prompt + "\n" + item.get("input")
                response = item.get("output", "")
                code = item.get(self.code_field, response)
                lang = item.get("language", "python")
            else:
                # Skip entries with unrecognized format
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
        """
        Filter out samples where the prompt alone exceeds max sequence length.

        This prevents two issues:
        1. Generating labels that are all -100 (no tokens to predict)
        2. Accidentally training on prompt completion instead of response generation
        """
        kept = []
        for it in self.data:
            # Tokenize prompt without special tokens to get accurate length
            prompt_enc = self.tokenizer(it["prompt"], add_special_tokens=False)
            if len(prompt_enc["input_ids"]) < self.max_seq_len:
                kept.append(it)
        self.data = kept

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def format_prompt(self, prompt: str) -> str:
        """
        Format a prompt with the required conversation template.

        Args:
            prompt: Raw prompt string

        Returns:
            Formatted prompt with user/assistant markers
        """
        # Simple template without special tokens (tokenizer handles special tokens separately)
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    def format_conversation(self, prompt: str, response: str) -> str:
        """
        Format a complete conversation including prompt and response.

        Args:
            prompt: User instruction/query
            response: Assistant's response/code generation

        Returns:
            Complete conversation string with EOS token if available
        """
        return self.format_prompt(prompt) + response + (self.tokenizer.eos_token or "")

    def __getitem__(self, idx):
        """
        Get a single training sample by index.

        This method:
        1. Formats the prompt and response into a conversation
        2. Tokenizes the conversation
        3. Creates labels with prompt tokens masked (-100)
        4. Pads sequences to max_seq_len
        5. Returns all necessary fields for training

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                input_ids: Tokenized conversation
                attention_mask: Attention mask for padding
                labels: Labels with prompt tokens masked
                language_id: Integer ID for the programming language
                code: Raw code string for AST evaluation
        """
        item = self.data[idx]

        # Build complete conversation text
        conversation = self.format_conversation(item["prompt"], item["response"])

        # Tokenize the conversation
        tokens = self.tokenizer(
            conversation,
            add_special_tokens=False,  # Special tokens handled separately
            max_length=self.max_seq_len,
            truncation=True,  # Truncate if exceeds max length
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Calculate prompt length for loss masking
        prompt_text = self.format_prompt(item["prompt"])
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        # Create labels tensor from input_ids
        labels = torch.tensor(input_ids)
        # Mask prompt tokens with -100 (ignored in loss computation)
        labels[:prompt_len] = -100
        labels = labels.clone()
        # Also mask padding tokens
        labels[tokens["attention_mask"] == 0] = -100

        # Pad sequences to max_seq_len if necessary
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels = torch.cat([labels, torch.full((pad_len,), -100)])

        # Convert language to integer ID for AST processing
        language = item.get("language", "python").lower()
        language_id = torch.tensor(self.lang2id.get(language, 0), dtype=torch.long)

        # Return all fields needed for training
        return {
            "input_ids": torch.tensor(input_ids),  # Token IDs
            "attention_mask": torch.tensor(attention_mask),  # Attention mask for padding
            "labels": labels,  # Labels with prompt masked
            "language_id": language_id,  # Language ID for AST
            "code": item.get("code")  # Raw code for AST evaluation
        }