import json
from human_eval.data import read_problems
from human_eval.evaluation import evaluate_functional_correctness

sample_file = "test/human_eval/models/Qwen3-14B_samples_pass@1.jsonl"

# k = [1] 表示只计算 pass@1
results = evaluate_functional_correctness(
    sample_file,
    k=[1],
    n_workers=8,
    timeout=3.0
)

print("\n====================")
print("Evaluation Results")
print("====================")

for key, value in results.items():
    print(f"{key}: {value}")