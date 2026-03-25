import os
from config import cfg
from data_collator import make_collate_fn

# Set HuggingFace endpoint to mirror site for improved accessibility in specific regions
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

from dataset import SFTDataset
from ast_utils import BatchASTProcessor
from utils import compute_fkl


class ALKDTrainer(Trainer):
    """
    Augmented Lagrangian Knowledge Distillation (ALKD) Trainer.

    This custom trainer extends HuggingFace's Trainer to implement knowledge distillation
    with AST (Abstract Syntax Tree) structural constraints using augmented Lagrangian
    optimization method.
    """

    def __init__(
            self,
            *args,
            ast_processor: BatchASTProcessor = None,
            teacher_model=None,
            temperature: float = 2.0,
            rho: float = 0.1,
            **kwargs
    ):
        """
        Initialize the ALKD Trainer.

        Args:
            *args: Variable length argument list passed to parent Trainer
            ast_processor: Processor for computing AST-based structural losses
            teacher_model: Pre-trained teacher model for knowledge distillation
            temperature: Temperature parameter for softening probability distributions
            rho: Penalty parameter for augmented Lagrangian method
            **kwargs: Arbitrary keyword arguments passed to parent Trainer
        """
        super().__init__(*args, **kwargs)

        # Store AST processor for structural loss computation
        self.ast_processor = ast_processor
        # Store teacher model for knowledge distillation
        self.teacher_model = teacher_model
        # Store temperature parameter for KL divergence computation
        self.temperature = temperature

        # Augmented Lagrangian parameters for constrained optimization
        self.lambda_ast = torch.tensor(0.0)  # Lagrange multiplier for AST constraint
        self.rho = rho  # Penalty parameter for quadratic penalty term

        # Store the most recent AST loss value for Lagrange multiplier update
        self.last_ast_loss = None

        # Mapping from language IDs to language names for AST processing
        self.lang_reverse_map = {
            0: "ruby",
            1: "rust",
            2: "javascript",
            3: "python",
            4: "julia",
            5: "typescript",
            6: "go",
            7: "c_sharp",
            8: "java",
            9: "cpp",
            10: "bash"
        }

    # ---------- Logits alignment for KL divergence computation ----------
    def align_logits(self, student_logits, teacher_logits, pad_value=-1e4):
        """
        Align teacher logits dimensions with student logits for KL computation.

        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            pad_value: Value used for padding when teacher vocabulary is smaller

        Returns:
            Aligned teacher logits matching student vocabulary size
        """
        s_vocab = student_logits.shape[-1]  # Student vocabulary size
        t_vocab = teacher_logits.shape[-1]  # Teacher vocabulary size

        # Case 1: Vocabulary sizes match - no alignment needed
        if t_vocab == s_vocab:
            return teacher_logits
        # Case 2: Teacher vocabulary larger - truncate to student size
        elif t_vocab > s_vocab:
            return teacher_logits[:, :, :s_vocab]
        # Case 3: Teacher vocabulary smaller - pad with specified value
        else:
            pad_size = s_vocab - t_vocab  # Calculate padding size
            pad_shape = list(teacher_logits.shape[:-1]) + [pad_size]
            pad_tensor = teacher_logits.new_full(pad_shape, pad_value)
            return torch.cat([teacher_logits, pad_tensor], dim=-1)

    # ---------- Batch code generation for AST evaluation ----------
    def generate_code_batch(self, model, inputs, max_new_tokens=256):
        """
        Generate code from model for a batch of inputs.

        Args:
            model: The model to use for generation
            inputs: Dictionary containing input_ids and attention_mask
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            List of decoded code strings
        """
        # Move inputs to model device
        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        # Generate outputs without gradient computation
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated token sequences to strings
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    # ---------- Total loss computation with KD and AST components ----------
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute total loss including knowledge distillation and AST constraints.

        Args:
            model: The student model
            inputs: Input data dictionary
            return_outputs: Whether to return model outputs along with loss

        Returns:
            Total loss value or tuple of (loss, outputs)
        """
        # Forward pass through student model
        outputs = model(**inputs)
        ce_loss = outputs.loss  # Cross-entropy loss from language modeling

        # ========== Knowledge Distillation Loss Computation ==========
        if self.teacher_model is not None:
            # Forward pass through teacher model (no gradients)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

            # Get logits from both models
            student_logits = outputs.logits
            # Align teacher logits to student vocabulary size
            teacher_logits = self.align_logits(student_logits, teacher_outputs.logits)

            # Compute forward KL divergence between student and teacher distributions
            fkl = compute_fkl(
                student_logits,
                teacher_logits,
                inputs.get("labels"),  # Labels for masking padding tokens
                padding_id=-100,  # Padding token identifier
                temp=self.temperature,  # Temperature for softening
            )

            # Combine distillation loss with cross-entropy loss
            kd_loss = 0.6 * fkl + 0.4 * ce_loss
        else:
            # Fall back to standard cross-entropy if no teacher provided
            kd_loss = ce_loss

        # ========== AST Structural Loss Computation ==========
        ast_loss = torch.tensor(
            0.0, dtype=kd_loss.dtype, device=kd_loss.device
        )

        ast_valid = False  # Flag indicating if AST computation succeeded

        # Compute AST loss if processor is available
        if self.ast_processor is not None:
            try:
                # Generate code from student model
                student_codes = self.generate_code_batch(model, inputs)
                # Get reference codes from inputs if available
                teacher_codes = inputs.get("code", None)
                languages = inputs.get("language_id", None)

                # Ensure teacher codes are properly formatted as strings
                if teacher_codes is not None:
                    teacher_codes = [
                        c if isinstance(c, str)
                        else c.decode("utf8") if isinstance(c, (bytes, bytearray))
                        else str(c)
                        for c in teacher_codes
                    ]
                else:
                    # If no reference codes, use student-generated codes as reference
                    teacher_codes = student_codes

                # Convert language IDs to language names
                if languages is not None and isinstance(languages, torch.Tensor):
                    lang_ids = languages.cpu().tolist()
                    languages = [
                        self.lang_reverse_map.get(i, "python")
                        for i in lang_ids
                    ]
                else:
                    # Default to Python if language information is missing
                    languages = ["python"] * len(teacher_codes)

                # Compute AST-based structural losses between teacher and student codes
                ast_losses = self.ast_processor.compute_batch_ast_loss(
                    teacher_codes,
                    student_codes,
                    languages
                )

                # Average losses across batch and move to correct device
                ast_loss = ast_losses.to(kd_loss.device).mean()
                ast_valid = True  # Mark AST computation as successful

            except Exception as e:
                # Handle any errors in AST computation gracefully
                print("AST loss failed:", e)
                ast_loss = torch.tensor(
                    0.0, dtype=kd_loss.dtype, device=kd_loss.device
                )
                ast_valid = False  # Mark AST computation as failed

        # ========== Augmented Lagrangian Constraint Handling ==========
        constraint = ast_loss  # Define constraint as AST loss

        total_loss = kd_loss  # Start with knowledge distillation loss

        # Apply augmented Lagrangian formulation if AST computation succeeded
        if ast_valid:
            total_loss = (
                    kd_loss
                    + self.lambda_ast.to(kd_loss.device) * constraint  # Linear term from Lagrange multiplier
                    + 0.5 * self.rho * constraint * constraint  # Quadratic penalty term
            )

            # Store current AST loss for Lagrange multiplier update
            self.last_ast_loss = constraint.detach()
        else:
            # Clear stored loss if AST computation failed
            self.last_ast_loss = None

        # Return loss with or without model outputs based on parameter
        return (total_loss, outputs) if return_outputs else total_loss

    # ---------- Lagrange multiplier update after each training step ----------
    def on_step_end(self, args, state, control, **kwargs):
        """
        Update Lagrange multiplier after each training step.

        This implements the multiplier update rule in augmented Lagrangian method:
        lambda = lambda + rho * constraint_value

        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kwargs: Additional keyword arguments

        Returns:
            Updated control object
        """
        # Update Lagrange multiplier if AST loss was computed
        if self.last_ast_loss is not None:
            self.lambda_ast = (
                    self.lambda_ast + self.rho * self.last_ast_loss.cpu()
            )

            # Apply clipping to prevent multiplier from growing too large
            self.lambda_ast = torch.clamp(self.lambda_ast, 0.0, 10.0)

        return control


# ==============================
# Main execution function
# ==============================
if __name__ == "__main__":

    # Disable tokenizer parallelism to avoid deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Load model names from configuration
    student_model_name = cfg.get("model", "student_model")
    teacher_model_name = cfg.get("model", "teacher_model")

    # Initialize tokenizer for student model
    tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        trust_remote_code=True  # Allow custom model code
    )

    # Set padding token if not already defined (use EOS token as fallback)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Student Model with LoRA Configuration --------
    # Load pre-trained student model
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        device_map="auto",  # Automatic device placement
        torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        use_cache=False  # Disable cache for gradient checkpointing compatibility
    )

    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # Required for gradient checkpointing

    # Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
    lora_config = LoraConfig(
        r=16,  # Rank of low-rank matrices
        lora_alpha=128,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention modules
        lora_dropout=0.05,  # Dropout probability for LoRA layers
        task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
        bias="none",  # Do not train bias parameters
    )

    # Apply LoRA configuration to model
    model = get_peft_model(model, lora_config)

    # -------- Teacher Model (Frozen) --------
    # Load pre-trained teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # Freeze all teacher model parameters
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()  # Set to evaluation mode

    # -------- Dataset Preparation --------
    # Initialize supervised fine-tuning dataset
    dataset = SFTDataset(
        data_path=cfg.get("data", "data_path"),  # Path to training data
        tokenizer=tokenizer,
        max_seq_len=1024,  # Maximum sequence length
    )

    # Initialize AST processor for structural loss computation
    ast_proc = BatchASTProcessor()

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results_al_kd",  # Directory for saving outputs
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size per device
        gradient_accumulation_steps=2,  # Steps for gradient accumulation
        bf16=True,  # Use bfloat16 precision
        learning_rate=2e-4,  # Learning rate
        warmup_ratio=0.03,  # Warmup ratio for learning rate scheduler
        logging_steps=50,  # Log every N steps
        save_steps=1000,  # Save checkpoint every N steps
        save_total_limit=2,  # Keep only last 2 checkpoints
    )

    # Create data collator for dynamic batching
    data_collator = make_collate_fn(tokenizer)

    # Initialize ALKD Trainer with all components
    trainer = ALKDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use same dataset for evaluation
        tokenizer=tokenizer,
        data_collator=data_collator,
        ast_processor=ast_proc,  # AST processor for structural constraints
        teacher_model=teacher_model,  # Teacher model for distillation
        temperature=2.0,  # Temperature for knowledge distillation
        rho=0.1,  # Initial penalty parameter for augmented Lagrangian
    )

    # Start training with augmented Lagrangian knowledge distillation
    print("Start Augmented Lagrangian KD Training")
    trainer.train()
