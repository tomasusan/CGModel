import pandas as pd
from collections import defaultdict

from ast_utils import ASTProcessor

language_type_map = {
    "C++": "cpp",
    "Java": "java",
    "Python": "python",
    "Ruby": "ruby",
    "C": "c",
    "C#": "c_sharp",
    "JavaScript": "javascript",
    "R": "r",
    "Bash": "bash",
    "Go": "go",
    "Julia": "julia",
    "TypeScript": "typescript",
    "Rust": "rust",
}


def validate_dataset(file_path: str, failed_output_path: str):
    df = pd.read_parquet(file_path)

    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    lang_stats = defaultdict(lambda: {"success": 0, "fail": 0})
    failed_samples = []

    # 缓存 processor（性能优化）
    processors = {}

    for idx, row in df.iterrows():
        code = row.get("response", None)
        raw_lang = row.get("programming_language", None)

        fail_reason = None

        # 1️⃣ 检查代码
        if not code or not str(code).strip():
            fail_reason = "missing_code"

        # 2️⃣ 检查语言字段
        elif not raw_lang:
            fail_reason = "missing_language"

        else:
            mapped_lang = language_type_map.get(raw_lang, None)

            # 3️⃣ 不支持语言
            if mapped_lang is None:
                fail_reason = "unsupported_language"

            else:
                # 4️⃣ 获取 processor
                if mapped_lang not in processors:
                    processors[mapped_lang] = ASTProcessor(mapped_lang)

                processor = processors[mapped_lang]

                # parser 初始化失败
                if processor.parser is None:
                    fail_reason = "parser_init_failed"
                else:
                    try:
                        tree = processor.code_to_ast(code)

                        if tree is None:
                            fail_reason = "parse_exception"
                        elif tree.root_node.has_error:
                            fail_reason = "syntax_error"
                        else:
                            # 成功
                            success_count += 1
                            lang_stats[raw_lang]["success"] += 1
                            continue

                    except Exception:
                        fail_reason = "parse_exception"

        # 如果走到这里说明失败
        fail_count += 1
        lang_stats[str(raw_lang)]["fail"] += 1

        failed_sample = row.to_dict()
        failed_sample["fail_reason"] = fail_reason
        failed_samples.append(failed_sample)

    # 输出统计
    print(f"AST Parse Success: {success_count}")
    print(f"AST Parse Fail   : {fail_count}")
    print(f"Success Rate     : {success_count / total_rows:.4f}")

    print("\nPer Language Statistics:")
    for lang, stats in lang_stats.items():
        total = stats["success"] + stats["fail"]
        rate = stats["success"] / total if total > 0 else 0
        print(
            f"{lang:12} | total: {total:6} | success: {stats['success']:6} | "
            f"fail: {stats['fail']:6} | success_rate: {rate:.4f}"
        )

    # 保存失败样本
    if failed_samples:
        failed_df = pd.DataFrame(failed_samples)
        failed_df.to_parquet(failed_output_path, index=False)
        print(f"\nFailed samples saved to: {failed_output_path}")
        print(f"Total failed samples: {len(failed_samples)}")
    else:
        print("\nNo failed samples.")


if __name__ == "__main__":
    file_path = "data/Output_data.parquet"
    failed_output_path = "data/failed_samples.parquet"
    validate_dataset(file_path, failed_output_path)

