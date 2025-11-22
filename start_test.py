import torch
import subprocess

if torch.cuda.is_available():
    print("CUDA可用")
    device_count = torch.cuda.device_count()
    print(f"📊 可用GPU数量: {device_count}")

    for i in range(device_count):
        print(f"\n--- GPU {i} 详细信息 ---")
        # 获取设备属性
        props = torch.cuda.get_device_properties(i)
        print(f"  设备名称: {props.name}")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  总显存: {props.total_memory / (1024 ** 3):.2f} GB")  # 转换为GB
        print(f"  多处理器数量: {props.multi_processor_count}")

        # 获取当前显存使用情况 (可选，需要额外的nvidia-smi调用)
        try:
            result = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits', '-i', str(i)
            ], encoding='utf-8')
            memory_used = int(result.strip())
            print(f"  已用显存: {memory_used} MB")
        except Exception as e:
            print(f"  无法获取已用显存: {e}")

else:
    print("CUDA不可用，将使用CPU")

print("检查tree-sitter兼容...")
from tree_sitter_languages import get_language, get_parser
parser = get_parser('cpp')
language = get_language('cpp')
if parser is None or language is None:
    print("tree-sitter兼容失败")
else:
    print("tree-sitter兼容检查成功")