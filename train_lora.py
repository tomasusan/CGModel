import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import SFTDataset
import json
from config import cfg


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
    # ==================== 配置参数 ====================
    base_model_name = cfg.get("model", "teacher_model")
    output_dir = base_model_name + "/lora/sft_full"
    data_path = cfg.get("data", "data_path")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("开始LoRA适配器训练 - 全精度版本")
    print("=" * 50)

    # ==================== 全精度加载模型 ====================
    print("1. 全精度加载模型...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        torch_dtype=torch.bfloat16,  # 使用bfloat16平衡精度和性能
    )

    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("模型加载完成")
    print_gpu_memory()

    # ==================== 增强的LoRA配置 ====================
    print("2. 配置增强LoRA...")

    lora_config = LoraConfig(
        r=16,  # 增加秩以获得更好性能
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "w1", "w2", "w3",  # 针对Qwen架构
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    print("LoRA配置完成")
    model.print_trainable_parameters()

    # ==================== 准备训练数据 ====================
    print("3. 准备训练数据...")

    if not os.path.exists(data_path):
        print(f"训练数据文件不存在: {data_path}")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"数据集加载: {len(data)} 条样本")

    # 使用更长的序列长度充分利用显存
    train_dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=2048  # 增加序列长度
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print(f"数据预处理完成，共 {len(train_dataset)} 条样本")

    # ==================== 优化的训练参数 ====================
    print("4. 配置训练参数...")

    training_args = TrainingArguments(
        # 输出配置
        output_dir=output_dir,
        overwrite_output_dir=True,

        # 训练周期和批次 - 充分利用80GB显存
        num_train_epochs=3,
        per_device_train_batch_size=8,  # 大幅增加batch size
        gradient_accumulation_steps=2,  # 减少梯度累积步数

        # 优化器参数
        learning_rate=2e-4,  # 提高学习率
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,

        # 精度和内存优化
        bf16=True,  # 使用bfloat16
        gradient_checkpointing=True,

        # 数据加载优化
        dataloader_pin_memory=True,
        dataloader_num_workers=4,

        # 保存和日志
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,
        eval_steps=500,
        evaluation_strategy="steps",

        # 其他参数
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        optim="adamw_torch",

        # 学习率调度
        lr_scheduler_type="cosine",
    )

    print("训练参数配置完成")

    # ==================== 创建训练器 ====================
    print("5. 创建训练器...")

    # 创建验证集（取前100条作为验证）
    eval_dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=2048
    )
    # 如果数据量大，可以取一部分作为验证集
    if len(eval_dataset) > 100:
        from torch.utils.data import Subset
        eval_indices = list(range(min(100, len(eval_dataset))))
        eval_dataset = Subset(eval_dataset, eval_indices)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("训练器创建完成")

    # ==================== 开始训练 ====================
    print("6. 开始训练...")
    print("训练前显存状态:")
    print_gpu_memory()

    # 检查可训练参数
    print("检查可训练参数...")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"可训练参数: {name}, 形状: {param.shape}")

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params / total_params * 100:.4f}%")

    # 开始训练
    train_result = trainer.train()

    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ==================== 保存模型 ====================
    print("7. 保存LoRA适配器...")

    # 保存最佳模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # 保存训练状态
    trainer.save_state()

    saved_files = os.listdir(output_dir)
    print(f"LoRA适配器已保存到: {output_dir}")
    print(f"保存的文件: {saved_files}")

    return output_dir


def validate_lora_adapter(lora_path):
    """
    验证LoRA适配器是否能正确加载
    """
    print("\n" + "=" * 50)
    print("验证LoRA适配器")
    print("=" * 50)

    try:
        # 全精度加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # 加载LoRA适配器
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()

        print("LoRA适配器验证成功！")

        # 测试推理
        tokenizer = AutoTokenizer.from_pretrained(lora_path)

        test_cases = [
            "写一个Python函数计算阶乘：",
            "用Java实现一个快速排序算法：",
            "写一个C++函数反转链表："
        ]

        for i, test_input in enumerate(test_cases):
            print(f"\n测试案例 {i + 1}: {test_input}")

            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成结果: {response[len(test_input):]}")

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