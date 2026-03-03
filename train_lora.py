from config import cfg
cfg.init_HuggingFace()

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
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"当前显存使用: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"最大显存使用: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        print(
            f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 - torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print()


def train_lora_adapter():

    base_model_name = cfg.get("model", "teacher_model")
    output_dir = f"lora/{base_model_name}_code_alignment"

    data_path = cfg.get("data", "data_path")

    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Configuring LoRA...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    print("Preparing dataset...")

    train_dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=2048,
    )

    print("Dataset size:", len(train_dataset))

    from torch.utils.data import Subset

    eval_size = min(200, len(train_dataset))
    eval_dataset = Subset(train_dataset, list(range(eval_size)))

    training_args = TrainingArguments(
        output_dir=output_dir,

        num_train_epochs=3,

        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,

        learning_rate=5e-5,
        weight_decay=0.01,

        warmup_steps=200,
        lr_scheduler_type="cosine",

        bf16=True,
        logging_steps=20,

        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,

        eval_strategy="steps",
        eval_steps=500,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        report_to="tensorboard",

        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        eval_dataset=eval_dataset,
    )

    print("Start training...")
    trainer.train()

    print("Saving LoRA...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Training finished.")

    return output_dir


def validate_lora_adapter(lora_path):
    """
    验证LoRA适配器是否能正确加载
    """
    print("\n" + "=" * 50)
    print("验证LoRA适配器")
    print("=" * 50)

    def build_prompt(prompt):
        return f"<|user|>\n{prompt}\n<|assistant|>\n"

    try:
        # 全精度加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.get("model", "teacher_model"),
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        # 加载LoRA适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()

        print("LoRA适配器验证成功！")

        # 测试推理
        tokenizer = AutoTokenizer.from_pretrained(lora_path)

        test_cases = [
            "write a Python code that can compute factorials：",
            "write a Java code that can do quick sort：",
            "write a C++ code that can reverse a list："
        ]

        for i, test_input in enumerate(test_cases):
            print(f"\n测试案例 {i + 1}: {test_input}")

            inputs = tokenizer(build_prompt(test_input), return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("生成结果:")
            print(response)

        return True

    except Exception as e:
        print(f"LoRA适配器验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 训练LoRA适配器
    lora_path = train_lora_adapter()

    if lora_path:
        # 验证适配器
        success = validate_lora_adapter(lora_path)

        if success:
            print("\n" + "=" * 50)
            print("LoRA适配器训练和验证完成！")
            print(f"适配器路径: '{lora_path}'")
            print("=" * 50)
        else:
            print("\nLoRA适配器训练完成但验证失败")
    else:
        print("\nLoRA适配器训练失败")