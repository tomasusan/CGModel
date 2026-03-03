import json
import torch
from config import cfg
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from human_eval.data import read_problems

base_model_path = cfg.get("model", 'teacher_model')
lora_path = f"lora/{base_model_path}_code_alignment"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.eval()

problems = read_problems()

print(len(problems), " problems generated")

results = []

from tqdm import tqdm

repeat = 1
print(f"generating results for pass@{repeat} for {base_model_path}")
for task_id, problem in tqdm(problems.items(), total=len(problems)):
    def build_prompt(in_prompt):
        return f"<|user|>\nsolve problem below, write code only:\n{in_prompt}\n<|assistant|>\n"

    def extract_code(in_text):
        if "<|assistant|>" in in_text:
            in_text = in_text.split("<|assistant|>")[-1]
        if "<|user|>" in in_text:
            in_text = in_text.split("<|user|>")[0]
        if "</think>" in in_text:
            in_text = in_text.replace("</think>", "")
        if "<think>" in in_text:
            in_text = in_text.replace("<think>", "")
        return in_text.strip()

    prompt = build_prompt(problem["prompt"])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = text[len(prompt):]
    completion = extract_code(completion)

    results.append({
        "task_id": task_id,
        "completion": completion
    })

from pathlib import Path
path = Path(f"test/human_eval/{base_model_path}_samples_pass@{repeat}.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建

with open(path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("生成完成")