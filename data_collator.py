import torch
from torch.nn.utils.rnn import pad_sequence

def make_collate_fn(tokenizer):
    tensor_fields = ["input_ids", "attention_mask", "labels", "language_id"]
    string_fields = ["code", "prompt"]   # 你可以随时增加

    def collate_fn(batch):
        result = {}

        # 1. 字符串字段：保持为列表，不做任何处理
        for key in string_fields:
            if key in batch[0]:
                result[key] = [sample[key] for sample in batch]

        # 2. Tensor 字段：使用 pad_sequence
        for key in tensor_fields:
            if key in batch[0]:
                tensors = [sample[key] for sample in batch]
                # input_ids 需要 pad_token_id，labels 要 -100
                if key == "input_ids":
                    pad_value = tokenizer.pad_token_id
                elif key == "labels":
                    pad_value = -100
                else:
                    pad_value = 0

                result[key] = pad_sequence(
                    tensors,
                    batch_first=True,
                    padding_value=pad_value
                )

        return result

    return collate_fn
