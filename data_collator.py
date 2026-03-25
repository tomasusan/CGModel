import torch
from torch.nn.utils.rnn import pad_sequence


def make_collate_fn(tokenizer):
    """
    Create a collate function for batching samples in a dataloader.

    This factory function returns a customized collate function that handles
    different types of fields appropriately:
    - String fields are kept as lists without modification
    - Tensor fields are padded to the maximum length in the batch

    Args:
        tokenizer: The tokenizer instance used for obtaining pad_token_id

    Returns:
        collate_fn: A function that collates a list of samples into a batch
    """

    # Define field categories for different processing strategies
    tensor_fields = ["input_ids", "attention_mask", "labels"]  # Fields that should be padded as tensors
    info_tensor_fields = ["language_id"]  # Additional tensor fields (currently unused but kept for extensibility)
    string_fields = ["code", "prompt"]  # Fields that remain as strings (can be extended as needed)

    def collate_fn(batch):
        """
        Collate a list of samples into a batched format.

        Args:
            batch: List of sample dictionaries, each containing various fields

        Returns:
            result: Dictionary containing batched data with appropriate formatting
        """
        result = {}

        # 1. Process string fields: keep as lists without any transformation
        for key in string_fields:
            if key in batch[0]:  # Check if field exists in the first sample
                # Extract the string values for this field from all samples
                result[key] = [sample[key] for sample in batch]

        # 2. Process tensor fields: pad sequences to the same length
        for key in tensor_fields:
            if key in batch[0]:  # Check if field exists in the first sample
                # Collect tensors for this field from all samples
                tensors = [sample[key] for sample in batch]

                # Determine appropriate padding value based on field type
                if key == "input_ids":
                    # Input IDs should be padded with tokenizer's pad token ID
                    pad_value = tokenizer.pad_token_id
                elif key == "labels":
                    # Labels should use -100 for padding (ignored in loss computation)
                    pad_value = -100
                else:
                    # Other tensors (e.g., attention_mask) use 0 for padding
                    pad_value = 0

                # Pad sequences to the maximum length in the batch
                result[key] = pad_sequence(
                    tensors,
                    batch_first=True,  # Return tensors with shape (batch, seq_len)
                    padding_value=pad_value  # Value to use for padding
                )

        return result

    return collate_fn