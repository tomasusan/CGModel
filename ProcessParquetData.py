import re

import pandas as pd


def clean_response_text(response):
    """
    清洗response文本，只保留代码块内容（包括注释），去除自然语言描述

    参数:
    response: 原始response字符串

    返回:
    清洗后的纯代码字符串（含注释），无代码块则返回空字符串
    """
    if pd.isna(response) or not isinstance(response, str):
        return ""

    # 正则表达式匹配代码块（支持```language ... ```格式）
    # 匹配规则：捕获```开头（可选语言）到```结尾之间的所有内容
    pattern = r'```[\w]*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)  # re.DOTALL让.匹配换行符

    if matches:
        # 合并所有匹配到的代码块（处理多代码块情况）
        code_content = '\n'.join(matches).strip()
        return code_content
    else:
        # 无代码块时返回空字符串（也可根据需求返回原内容）
        return ""


def process_parquet_data(file_path, output_path, sample_size=10000):
    """
    处理parquet数据集，按要求筛选和采样数据

    参数:
    file_path: parquet文件路径
    output_path: 输出文件路径
    sample_size: 最终采样数量，默认10000
    """
    # 1. 读取parquet文件
    print("正在读取数据...")
    df = pd.read_parquet(file_path)

    # 2. 筛选指定列
    print("正在筛选列...")
    target_columns = ['prompt', 'adjective', 'programming_language', 'response']
    # 确保列存在，避免KeyError
    existing_columns = [col for col in target_columns if col in df.columns]
    df_filtered = df[existing_columns].copy()

    # 3. 过滤编程语言
    print("正在过滤编程语言...")
    exclude_languages = ['Neo4j database and Cypher', 'relation database and SQL']
    df_filtered = df_filtered[~df_filtered['programming_language'].isin(exclude_languages)]

    # 4. 清洗response列（核心新增步骤）
    print("正在清洗response列，提取纯代码...")
    df_filtered['response'] = df_filtered['response'].apply(clean_response_text)

    # 过滤掉清洗后response为空的行（可选，根据需求决定是否保留）
    df_filtered = df_filtered[df_filtered['response'] != ""]

    # 检查是否有难度列（根据你的描述，难度列是adjective）
    if 'adjective' not in df_filtered.columns:
        raise ValueError("数据中未找到难度列（adjective），请检查列名是否正确")

    # 5. 定义难度比例
    adjective_proportions = {
        'Low': 0.336,
        'High': 0.334,
        'Extreme': 0.330
    }

    # 6. 按编程语言分组，然后按难度比例采样
    print("正在按比例采样数据...")
    sampled_dfs = []

    # 获取所有需要保留的编程语言列表
    languages = df_filtered['programming_language'].unique()

    # 处理空语言列表的情况
    if len(languages) == 0:
        raise ValueError("过滤后无有效编程语言数据，请检查过滤条件")

    # 计算每种语言应采样的基础数量（平均分配）
    lang_sample_base = sample_size / len(languages)

    for lang in languages:
        # 筛选当前语言的数据
        lang_df = df_filtered[df_filtered['programming_language'] == lang].copy()

        # 对当前语言按难度比例采样
        lang_sampled = []
        for adjective, proportion in adjective_proportions.items():
            # 筛选当前难度的数据
            diff_df = lang_df[lang_df['adjective'] == adjective].copy()

            # 计算该难度应采样的数量
            diff_sample_size = int(round(lang_sample_base * proportion))

            # 如果数据量不足，则取全部
            if len(diff_df) <= diff_sample_size:
                sampled = diff_df
            else:
                sampled = diff_df.sample(n=diff_sample_size, random_state=42)  # 设置随机种子保证可复现

            lang_sampled.append(sampled)

        # 合并当前语言的所有难度采样数据
        lang_sampled_df = pd.concat(lang_sampled, ignore_index=True)
        sampled_dfs.append(lang_sampled_df)

    # 合并所有采样数据
    final_df = pd.concat(sampled_dfs, ignore_index=True)

    # 7. 调整最终数量到精确的1万条（可能因四舍五入有微小偏差）
    if len(final_df) > sample_size:
        final_df = final_df.sample(n=sample_size, random_state=42)
    elif len(final_df) < sample_size:
        print(f"警告：符合条件的数据不足{sample_size}条，实际采样{len(final_df)}条")

    # 8. 保存结果
    print(f"正在保存结果，最终数据量：{len(final_df)}条")
    final_df.to_parquet(output_path, index=False)
    # 也可以保存为CSV格式（如果需要）
    # final_df.to_csv(output_path.replace('.parquet', '.csv'), index=False, encoding='utf-8')

    print(f"数据处理完成！结果已保存至: {output_path}")

    # 输出采样统计信息
    print("\n采样结果统计：")
    print("=" * 50)
    print("编程语言分布：")
    print(final_df['programming_language'].value_counts())
    print("\n难度（adjective）分布：")
    print(final_df['adjective'].value_counts(normalize=True).round(4) * 100)

    return final_df


# 主程序执行
if __name__ == "__main__":
    # 请修改以下路径为你的实际文件路径
    input_file = "part_1_200000.parquet"  # 输入parquet文件路径
    output_file = "data/Output_data.parquet"  # 输出文件路径

    try:
        # 执行数据处理
        result_df = process_parquet_data(input_file, output_file, sample_size=10000)
    except Exception as e:
        print(f"处理过程中出错：{e}")