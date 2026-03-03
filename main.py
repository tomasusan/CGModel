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
from utils import compute_fkl


class ALKDTrainer(Trainer):
    def __init__(
        self,
        *args,
        ast_processor: BatchASTProcessor = None,
        teacher_model=None,
        temperature: float = 2.0,
        rho: float = 0.1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.ast_processor = ast_processor
        self.teacher_model = teacher_model
        self.temperature = temperature

        # Augmented Lagrangian parameters
        self.lambda_ast = torch.tensor(0.0)
        self.rho = rho

        self.last_ast_loss = None

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

    # ---------- KL对齐 ----------
    def align_logits(self, student_logits, teacher_logits, pad_value=-1e4):
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

    # ---------- code generation ----------
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

    # ---------- Loss-Total ----------
    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        ce_loss = outputs.loss

        # ========== KD LOSS ==========
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

            student_logits = outputs.logits
            teacher_logits = self.align_logits(student_logits, teacher_outputs.logits)

            fkl = compute_fkl(
                student_logits,
                teacher_logits,
                inputs.get("labels"),
                padding_id=-100,
                temp=self.temperature,
            )

            kd_loss = 0.6 * fkl + 0.4 * ce_loss
        else:
            kd_loss = ce_loss

        # ========== AST LOSS ==========
        ast_loss = torch.tensor(
            0.0, dtype=kd_loss.dtype, device=kd_loss.device
        )

        ast_valid = False

        if self.ast_processor is not None:
            try:
                student_codes = self.generate_code_batch(model, inputs)
                teacher_codes = inputs.get("code", None)
                languages = inputs.get("language_id", None)

                if teacher_codes is not None:
                    teacher_codes = [
                        c if isinstance(c, str)
                        else c.decode("utf8") if isinstance(c, (bytes, bytearray))
                        else str(c)
                        for c in teacher_codes
                    ]
                else:
                    teacher_codes = student_codes

                if languages is not None and isinstance(languages, torch.Tensor):
                    lang_ids = languages.cpu().tolist()
                    languages = [
                        self.lang_reverse_map.get(i, "python")
                        for i in lang_ids
                    ]
                else:
                    languages = ["python"] * len(teacher_codes)

                ast_losses = self.ast_processor.compute_batch_ast_loss(
                    teacher_codes,
                    student_codes,
                    languages
                )

                ast_loss = ast_losses.to(kd_loss.device).mean()
                ast_valid = True

            except Exception as e:
                print("AST loss failed:", e)
                ast_loss = torch.tensor(
                    0.0, dtype=kd_loss.dtype, device=kd_loss.device
                )
                ast_valid = False

        # ========== AUGMENTED LAGRANGIAN ==========
        constraint = ast_loss

        total_loss = kd_loss

        if ast_valid:
            total_loss = (
                kd_loss
                + self.lambda_ast.to(kd_loss.device) * constraint
                + 0.5 * self.rho * constraint * constraint
            )

            self.last_ast_loss = constraint.detach()
        else:
            self.last_ast_loss = None

        return (total_loss, outputs) if return_outputs else total_loss

    # ---------- Update lambda ----------
    def on_step_end(self, args, state, control, **kwargs):
        if self.last_ast_loss is not None:
            self.lambda_ast = (
                self.lambda_ast + self.rho * self.last_ast_loss.cpu()
            )

            # 防止爆炸
            self.lambda_ast = torch.clamp(self.lambda_ast, 0.0, 10.0)

        return control


# ==============================
# 主函数
# ==============================
if __name__ == "__main__":

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    student_model_name = cfg.get("model", "student_model")
    teacher_model_name = cfg.get("model", "teacher_model")

    tokenizer = AutoTokenizer.from_pretrained(
        student_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Student + LoRA --------
    model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # -------- Teacher --------
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    # -------- Dataset --------
    dataset = SFTDataset(
        data_path=cfg.get("data", "data_path"),
        tokenizer=tokenizer,
        max_seq_len=1024,
    )

    ast_proc = BatchASTProcessor()

    training_args = TrainingArguments(
        output_dir="./results_al_kd",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        bf16=True,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=2,
    )

    data_collator = make_collate_fn(tokenizer)

    trainer = ALKDTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        ast_processor=ast_proc,
        teacher_model=teacher_model,
        temperature=2.0,
        rho=0.1,
    )

    print("Start Augmented Lagrangian KD Training")
    trainer.train()