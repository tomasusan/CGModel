from config import cfg

cfg.init_HuggingFace()  # Initialize HuggingFace environment with cache and mirror settings

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, TaskType
from dataset import SFTDataset


def print_gpu_memory():
    """
    Print current GPU memory usage information for monitoring.

    This function displays:
    - GPU device name
    - Current allocated memory
    - Maximum allocated memory (peak)
    - Available memory
    """
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 - torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print()


def train_lora_adapter():
    """
    Train a LoRA (Low-Rank Adaptation) adapter for code generation fine-tuning.

    This function:
    1. Loads a base model from HuggingFace
    2. Configures LoRA for efficient fine-tuning
    3. Prepares the SFT dataset
    4. Sets up training arguments
    5. Trains the LoRA adapter
    6. Saves the adapter weights

    Returns:
        Path to the saved LoRA adapter
    """
    # Get configuration parameters
    base_model_name = cfg.get("model", "teacher_model")
    output_dir = f"lora/{base_model_name}_code_alignment"
    data_path = cfg.get("data", "data_path")

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")

    # Load base model with memory-efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="auto",  # Automatic device placement
        trust_remote_code=True,  # Allow custom model code
        use_cache=False,  # Disable cache for gradient checkpointing
        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
    )

    # Enable gradient checkpointing to reduce memory usage during training
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # Required for gradient checkpointing

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # Set padding token if not already defined (use EOS token as fallback)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring LoRA...")

    # Configure LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",  # Query projection in attention
            "k_proj",  # Key projection in attention
            "v_proj",  # Value projection in attention
            "o_proj",  # Output projection in attention
        ],
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        bias="none",  # Do not train bias parameters
        task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
    )

    # Apply LoRA configuration to model
    model = get_peft_model(model, lora_config)

    print("Preparing dataset...")

    # Load training dataset
    train_dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=2048,
    )

    print("Dataset size:", len(train_dataset))

    # Create evaluation dataset (subset of training data)
    from torch.utils.data import Subset
    eval_size = min(200, len(train_dataset))
    eval_dataset = Subset(train_dataset, list(range(eval_size)))

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory for saving checkpoints

        # Training epochs
        num_train_epochs=3,

        # Batch size and gradient accumulation
        per_device_train_batch_size=2,  # Batch size per GPU
        gradient_accumulation_steps=16,  # Accumulate gradients for effective larger batch

        # Optimization settings
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay for regularization

        # Learning rate schedule
        warmup_steps=200,  # Number of warmup steps
        lr_scheduler_type="cosine",  # Cosine learning rate schedule

        # Precision and logging
        bf16=True,  # Use bfloat16 precision
        logging_steps=20,  # Log every N steps

        # Checkpoint saving
        save_strategy="steps",
        save_steps=1000,  # Save every 1000 steps
        save_total_limit=2,  # Keep only last 2 checkpoints

        # Evaluation settings
        eval_strategy="steps",
        eval_steps=500,  # Evaluate every 500 steps

        # Model selection
        load_best_model_at_end=True,  # Load best model at end of training
        metric_for_best_model="eval_loss",  # Use eval loss for model selection

        # DataLoader settings
        dataloader_num_workers=4,  # Number of data loading workers
        dataloader_pin_memory=True,  # Pin memory for faster GPU transfer

        # Logging
        report_to="tensorboard",  # Log to TensorBoard

        # Gradient clipping
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Maximum gradient norm for clipping
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,  # Default collator for batching
        eval_dataset=eval_dataset,
    )

    print("Start training...")
    trainer.train()

    print("Saving LoRA...")
    trainer.save_model()  # Save model weights
    tokenizer.save_pretrained(output_dir)  # Save tokenizer

    print("Training finished.")

    return output_dir


def validate_lora_adapter(lora_path):
    """
    Validate that the LoRA adapter can be loaded correctly and perform inference.

    This function:
    1. Loads the base model
    2. Loads the LoRA adapter
    3. Tests inference on sample prompts

    Args:
        lora_path: Path to the saved LoRA adapter

    Returns:
        Boolean indicating whether validation succeeded
    """
    print("\n" + "=" * 50)
    print("Validating LoRA Adapter")
    print("=" * 50)

    def build_prompt(prompt):
        """Build a formatted prompt for code generation."""
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    try:
        # Load base model in full precision
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.get("model", "teacher_model"),
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        # Load LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()  # Set to evaluation mode

        print("LoRA adapter validation successful!")

        # Test inference with sample prompts
        tokenizer = AutoTokenizer.from_pretrained(lora_path)

        # Define test cases for different programming languages
        test_cases = [
            "write a Python code that can compute factorials：",
            "write a Java code that can do quick sort：",
            "write a C++ code that can reverse a list："
        ]

        # Run inference for each test case
        for i, test_input in enumerate(test_cases):
            print(f"\nTest Case {i + 1}: {test_input}")

            # Tokenize input
            inputs = tokenizer(build_prompt(test_input), return_tensors="pt").to(model.device)

            # Generate response without gradient computation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,  # Maximum new tokens to generate
                    num_return_sequences=1,  # Return single sequence
                    temperature=0.7,  # Temperature for sampling
                    do_sample=True,  # Enable sampling (non-greedy)
                    pad_token_id=tokenizer.eos_token_id,  # Padding token
                    eos_token_id=tokenizer.eos_token_id  # End-of-sequence token
                )

            # Decode and display response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Generated output:")
            print(response)

        return True

    except Exception as e:
        print(f"LoRA adapter validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Main execution block
if __name__ == "__main__":
    # Train LoRA adapter
    lora_path = train_lora_adapter()

    if lora_path:
        # Validate the trained adapter
        success = validate_lora_adapter(lora_path)

        if success:
            print("\n" + "=" * 50)
            print("LoRA adapter training and validation completed successfully!")
            print(f"Adapter path: '{lora_path}'")
            print("=" * 50)
        else:
            print("\nLoRA adapter training completed but validation failed")
    else:
        print("\nLoRA adapter training failed")