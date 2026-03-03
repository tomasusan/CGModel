import re
import pandas as pd

# =========================
# 语言映射
# =========================
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

# =========================
# 代码清洗函数
# =========================
def clean_response_text(response):
    if pd.isna(response) or not isinstance(response, str):
        return ""

    pattern = r'```[\w]*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return '\n'.join(matches).strip()
    return ""


# =========================
# 不同语言的包装函数
# =========================
def wrap_code_if_needed(code: str, lang: str):
    if lang == "c_sharp":
        return f"""
class Program
{{
    static void Main(string[] args)
    {{
        {code}
    }}
}}
"""
    elif lang == "java":
        return f"""
class Main {{
    public static void main(String[] args) {{
        {code}
    }}
}}
"""
    elif lang in ["c", "cpp"]:
        return f"""
int main() {{
    {code}
    return 0;
}}
"""
    elif lang == "go":
        return f"""
package main
func main() {{
    {code}
}}
"""
    else:
        return code


from ast_utils import ASTProcessor

# =========================
# AST 校验函数
# =========================
def filter_valid_ast_rows(df):
    print("正在进行 AST 校验...")

    processors = {}
    valid_mask = []

    for idx, row in df.iterrows():
        code = row["response"]
        raw_lang = row["programming_language"]

        mapped_lang = language_type_map.get(raw_lang)

        # 语言不支持
        if mapped_lang is None:
            valid_mask.append(False)
            continue

        # 缓存 processor（避免重复创建）
        if mapped_lang not in processors:
            processors[mapped_lang] = ASTProcessor(mapped_lang)

        processor = processors[mapped_lang]

        # parser 初始化失败
        if processor.parser is None:
            valid_mask.append(False)
            continue

        try:
            tree = processor.code_to_ast(code)

            if tree is None:
                valid_mask.append(False)
            elif tree.root_node.has_error:
                valid_mask.append(False)
            else:
                valid_mask.append(True)

        except Exception:
            valid_mask.append(False)

    df["ast_valid"] = valid_mask
    df_valid = df[df["ast_valid"]].drop(columns=["ast_valid"])

    print(f"AST 过滤前: {len(df)}")
    print(f"AST 过滤后: {len(df_valid)}")
    print(f"AST 成功率: {len(df_valid) / len(df):.4f}")

    return df_valid


# =========================
# 主处理函数
# =========================
def process_parquet_data(file_path, output_path, sample_size=10000):

    print("正在读取数据...")
    df = pd.read_parquet(file_path)

    print("正在筛选列...")
    target_columns = ['prompt', 'adjective', 'programming_language', 'response']
    existing_columns = [col for col in target_columns if col in df.columns]
    df_filtered = df[existing_columns].copy()

    print("正在过滤编程语言...")
    exclude_languages = ['Neo4j database and Cypher', 'relation database and SQL']
    df_filtered = df_filtered[~df_filtered['programming_language'].isin(exclude_languages)]

    print("正在清洗response列...")
    df_filtered['response'] = df_filtered['response'].apply(clean_response_text)
    df_filtered = df_filtered[df_filtered['response'] != ""]

    # =========================
    # 新增 AST 校验步骤
    # =========================
    df_filtered = filter_valid_ast_rows(df_filtered)

    if 'adjective' not in df_filtered.columns:
        raise ValueError("数据中未找到难度列（adjective）")

    adjective_proportions = {
        'Low': 0.336,
        'High': 0.334,
        'Extreme': 0.330
    }

    print("正在按比例采样数据...")
    sampled_dfs = []
    languages = df_filtered['programming_language'].unique()

    lang_sample_base = sample_size / len(languages)

    for lang in languages:
        lang_df = df_filtered[df_filtered['programming_language'] == lang].copy()
        lang_sampled = []

        for adjective, proportion in adjective_proportions.items():
            diff_df = lang_df[lang_df['adjective'] == adjective].copy()
            diff_sample_size = int(round(lang_sample_base * proportion))

            if len(diff_df) <= diff_sample_size:
                sampled = diff_df
            else:
                sampled = diff_df.sample(n=diff_sample_size, random_state=42)

            lang_sampled.append(sampled)

        sampled_dfs.append(pd.concat(lang_sampled, ignore_index=True))

    final_df = pd.concat(sampled_dfs, ignore_index=True)

    if len(final_df) > sample_size:
        final_df = final_df.sample(n=sample_size, random_state=42)

    print(f"最终数据量: {len(final_df)}")
    final_df.to_parquet(output_path, index=False)

    print("数据处理完成！")

    return final_df


if __name__ == "__main__":
    input_file = "part_1_200000.parquet"
    output_file = "data/Output_data.parquet"

    process_parquet_data(input_file, output_file, sample_size=10000)
