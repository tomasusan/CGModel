import os
from config import cfg
from data_collator import make_collate_fn

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


class KGTrainer(Trainer):
    def __init__(self, *args, ast_processor: BatchASTProcessor = None, ast_loss_weight: float = 0.0, teacher_model=None, if_use_entropy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ast_processor = ast_processor
        self.ast_loss_weight = ast_loss_weight
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self.lang_reverse_map = {
            0: "ruby",
            1: "rust",
            2: "javascript",
            3: "python",
            4: "julia",
            5: "typescript",
            6: "go",
            7: "c#",
            8: "java",
            9: "cpp",
            10: "bash"
        }

    def align_logits(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, pad_value: float = -1e4):
        s_vocab = student_logits.shape[-1]
        t_vocab = teacher_logits.shape[-1]
        if t_vocab == s_vocab:
            return teacher_logits
        elif t_vocab > s_vocab:
            return teacher_logits[:, :, :s_vocab]
        else:
            pad_size = s_vocab - t_vocab
            pad_shape = list(teacher_logits.shape[:-1]) + [pad_size]
            pad_tensor = teacher_logits.new_full(pad_shape, pad_value)
            return torch.cat([teacher_logits, pad_tensor], dim=-1)

    def generate_code_batch(self, model, inputs, max_new_tokens=256):
        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        # teacher logits (no grad)
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
            teacher_logits = teacher_outputs.logits
            student_logits = outputs.logits
            teacher_logits = self.align_logits(student_logits, teacher_logits)

            # compute KL (you should ensure compute_fkl expects logits)
            from utils import compute_fkl, compute_rkl
            fkl = compute_fkl(student_logits, teacher_logits, inputs.get("labels"), padding_id=-100, temp=2.0)
            rkl = compute_rkl(outputs.logits, teacher_logits, inputs.get("labels"), padding_id=-100, temp=2.0)

            kd_loss = 0.7 * fkl + 0.3 * rkl
            if self.if_use_entropy:
                loss_total = 0.6 * kd_loss + 0.4 * loss
            else:
                loss_total = kd_loss
        else:
            loss_total = loss

        # AST loss (optional)
        ast_loss = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        if self.ast_processor is not None and self.ast_loss_weight > 0.0:
            try:
                # generate student codes
                student_codes = self.generate_code_batch(model, inputs)
                # teacher codes: prefer a 'code' field in dataset
                teacher_codes = inputs.get("code", None)
                # languages
                languages = inputs.get("language_id", None)

                # coerce teacher_codes to list of strings if it's a tensor/bytes
                if teacher_codes is not None:
                    teacher_codes = [c if isinstance(c, str) else c.decode("utf8") if isinstance(c, (bytes, bytearray)) else str(c) for c in teacher_codes]
                else:
                    # fallback: use decoded labels as teacher code
                    teacher_codes = student_codes  # at least avoid crash

                # languages to strings
                if languages is not None:
                    if isinstance(languages, torch.Tensor):
                        lang_ids = languages.cpu().tolist()
                        # Inverse mapping is required in main; here assume Trainer.dataset.lang2id exists
                        inv_map = self.lang_reverse_map
                        languages = [inv_map.get(i, "python") if inv_map else "python" for i in lang_ids]
                else:
                    languages = [None] * len(teacher_codes)

                ast_losses = self.ast_processor.compute_batch_ast_loss(teacher_codes, student_codes, languages)
                ast_loss = ast_losses.to(loss_total.device).mean()

            except Exception as e:
                print("AST loss computation failed:", e)
                ast_loss = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

        total = loss_total + self.ast_loss_weight * ast_loss
        if return_outputs:
            return total, outputs
        return total


if __name__ == "__main__":
    # --- minimal example to launch training ---
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    student_model_name = cfg.get("model", "student_model")
    teacher_model_name = cfg.get("model", "teacher_model")

    tokenizer = AutoTokenizer.from_pretrained(student_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load student model and apply LoRA
    model = AutoModelForCausalLM.from_pretrained(student_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, use_cache=False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # teacher (no grad)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    # dataset
    dataset = SFTDataset(
        data_path= cfg.get("data", "data_path"),
        tokenizer=tokenizer,
        max_seq_len=1024,
        lang2id={"ruby": 0,
                 "rust": 1,
                 "javascript": 2,
                 "python": 3,
                 "julia": 4,
                 "typescript": 5,
                 "go": 6,
                 "c#": 7,
                 "java": 8,
                 "c++": 9,
                 "bash": 10,
        }
    )

    ast_proc = BatchASTProcessor()

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        bf16=True,
        logging_steps=50,
        save_steps=1000,
    )

    data_collator = make_collate_fn(tokenizer)

    trainer = KGTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        ast_processor=ast_proc,
        ast_loss_weight=0.3,
        teacher_model=teacher_model,
        if_use_entropy=True,
    )

    print("Start training")
    trainer.train()

