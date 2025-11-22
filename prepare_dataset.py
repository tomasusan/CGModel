import pandas as pd
import json
import os
import re
from collections import defaultdict


class CodeDataProcessor:
    def __init__(self, parquet_path):
        """初始化数据处理器"""
        self.parquet_path = parquet_path
        self.df = None
        self.load_data()

    def load_data(self):
        """加载Parquet数据"""
        print(f"正在加载数据: {self.parquet_path}")
        self.df = pd.read_parquet(self.parquet_path)
        print(f"数据加载完成，总条数: {len(self.df):,}")

        # 显示基本信息
        self.show_basic_info()

    def show_basic_info(self):
        """显示基本信息"""
        print("\n" + "=" * 50)
        print("数据集基本信息")
        print("=" * 50)

        # 显示所有语言
        languages = self.df['programming_language'].value_counts()
        print(f"所有编程语言 ({len(languages)} 种):")
        for lang, count in languages.items():
            print(f"  {lang}: {count:,} 条")

        # 显示难度分布
        print(f"\n难度分布:")
        difficulty_dist = self.df['adjective'].value_counts()
        for level, count in difficulty_dist.items():
            print(f"  {level}: {count:,} 条 ({count / len(self.df) * 100:.1f}%)")

    def extract_pure_code(self, text):
        """提取 ``` 和 ``` 之间的纯代码"""
        if pd.isna(text):
            return ""

        text_str = str(text)

        # 使用正则表达式提取代码块
        pattern = r'```(?:\w+)?\s*(.*?)```'
        matches = re.findall(pattern, text_str, re.DOTALL)

        if matches:
            # 返回第一个代码块，去除前后空白
            code = matches[0].strip()
            return code
        else:
            # 如果没有代码块标记，返回原始文本
            return text_str.strip()

    def process_all_responses(self):
        """处理所有response字段，提取纯代码"""
        print("\n正在提取纯代码...")

        # 复制一份数据，避免修改原数据
        processed_df = self.df.copy()

        # 应用代码提取
        processed_df['pure_code'] = processed_df['response'].apply(self.extract_pure_code)

        # 移除提取后为空的内容
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df['pure_code'].str.len() > 0]
        final_count = len(processed_df)

        print(f"代码提取完成: {initial_count:,} → {final_count:,} 条")

        return processed_df

    def filter_quality_data(self, df):
        """筛选高质量数据"""
        print("\n正在筛选高质量数据...")

        initial_count = len(df)

        # 基本质量筛选
        df = df[
            (df['pure_code'].str.len() >= 30) &  # 代码长度至少30字符
            (df['pure_code'].str.len() <= 5000) &  # 代码长度最多5000字符
            (df['prompt'].str.len() >= 10) &  # 提示长度至少10字符
            (~df['pure_code'].str.contains(r'^\s*$'))  # 不是纯空白
            ]

        final_count = len(df)
        print(f"质量筛选完成: {initial_count:,} → {final_count:,} 条")

        return df

    def create_balanced_dataset(self, df, target_size=20000):
        """创建平衡的数据集"""
        print(f"\n开始创建平衡数据集，目标大小: {target_size:,}")

        # 定义要排除的语言
        excluded_languages = [
            "Neo4j database and Cypher",
            "relation database and SQL"
        ]

        # 过滤掉要排除的语言
        filtered_df = df[~df['programming_language'].isin(excluded_languages)]

        print(f"排除语言: {excluded_languages}")
        print(f"过滤后数据量: {len(df):,} → {len(filtered_df):,} 条")

        # 难度权重配置
        difficulty_weights = {
            'Low': 0.25,  # 25%
            'High': 0.50,  # 50%
            'Extreme': 0.25  # 25%
        }

        # 获取所有语言（排除后的）
        all_languages = filtered_df['programming_language'].unique()
        print(f"可用的编程语言: {len(all_languages)} 种")

        # 计算每种语言应该采样的总数
        samples_per_language = target_size // len(all_languages)
        remaining_samples = target_size % len(all_languages)

        print(f"基础采样: 每种语言 {samples_per_language} 条")
        if remaining_samples > 0:
            print(f"额外采样: {remaining_samples} 条将分配给数据量较多的语言")

        # 为每种语言收集样本
        all_samples = []

        for language in all_languages:
            print(f"\n处理语言: {language}")

            # 获取该语言的所有数据
            lang_data = filtered_df[filtered_df['programming_language'] == language]

            if len(lang_data) == 0:
                print(f"  没有找到 {language} 的数据")
                continue

            # 计算该语言每种难度的目标数量
            lang_target_size = samples_per_language
            # 为前几个语言分配额外的样本
            if remaining_samples > 0 and len(lang_data) >= lang_target_size + 1:
                lang_target_size += 1
                remaining_samples -= 1

            lang_samples = self.sample_by_difficulty(lang_data, lang_target_size, difficulty_weights)

            if len(lang_samples) > 0:
                all_samples.append(lang_samples)
                print(f"  采集 {len(lang_samples)} 条样本")
            else:
                print(f"  {language} 没有足够的合格样本")

        # 合并所有样本
        if all_samples:
            balanced_df = pd.concat(all_samples, ignore_index=True)

            # 如果总数超过目标，随机抽样到目标大小
            if len(balanced_df) > target_size:
                balanced_df = balanced_df.sample(n=target_size, random_state=42)

            return balanced_df
        else:
            print("错误: 没有采集到任何样本")
            return pd.DataFrame()

    def sample_by_difficulty(self, lang_data, target_size, difficulty_weights):
        """按难度权重从一种语言中采样"""
        samples = []

        for difficulty, weight in difficulty_weights.items():
            # 计算该难度的目标数量
            difficulty_target = int(target_size * weight)

            # 获取该难度的数据
            difficulty_data = lang_data[lang_data['adjective'] == difficulty]

            if len(difficulty_data) == 0:
                print(f"    {difficulty} 难度没有数据")
                continue

            # 采样
            if len(difficulty_data) >= difficulty_target:
                sampled = difficulty_data.sample(n=difficulty_target, random_state=42)
            else:
                sampled = difficulty_data
                print(f"    {difficulty} 难度只有 {len(difficulty_data)} 条，少于目标 {difficulty_target}")

            samples.append(sampled)

        if samples:
            return pd.concat(samples, ignore_index=True)
        else:
            return pd.DataFrame()

    def analyze_final_distribution(self, df):
        """分析最终数据分布"""
        print("\n" + "=" * 50)
        print("最终数据集分布分析")
        print("=" * 50)

        print(f"总数据量: {len(df):,} 条")

        # 显示排除的语言信息
        excluded_languages = ["Neo4j database and Cypher", "relation database and SQL"]
        print(f"排除的语言: {excluded_languages}")

        # 整体难度分布
        print(f"\n整体难度分布:")
        total_by_difficulty = df['adjective'].value_counts()
        total_samples = len(df)
        for level, count in total_by_difficulty.items():
            percentage = count / total_samples * 100
            print(f"  {level}: {count} 条 ({percentage:.1f}%)")

        # 按语言分组的难度分布
        print(f"\n按语言分组的难度分布:")
        language_groups = df.groupby('programming_language')

        for language, group in language_groups:
            lang_total = len(group)
            print(f"\n{language} ({lang_total} 条):")

            difficulty_counts = group['adjective'].value_counts()
            for level in ['Low', 'High', 'Extreme']:
                if level in difficulty_counts:
                    count = difficulty_counts[level]
                    percentage = count / lang_total * 100
                    print(f"  {level}: {count} 条 ({percentage:.1f}%)")
                else:
                    print(f"  {level}: 0 条 (0.0%)")

        # 代码长度统计
        print(f"\n代码长度统计:")
        code_lengths = df['pure_code'].str.len()
        print(f"  平均长度: {code_lengths.mean():.0f} 字符")
        print(f"  最小长度: {code_lengths.min()} 字符")
        print(f"  最大长度: {code_lengths.max()} 字符")

        # 显示代码提取示例
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"\n代码提取示例:")
            print(f"  语言: {sample['programming_language']}")
            print(f"  难度: {sample['adjective']}")
            print(f"  Prompt: {sample['prompt'][:100]}...")
            print(f"  代码前100字符: {sample['pure_code'][:100]}...")

    def save_dataset(self, df, output_path, format='parquet'):
        """保存数据集"""
        print(f"\n保存数据集到: {output_path}")

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        if format == 'parquet':
            # 保存处理后的数据
            save_df = df[['prompt', 'adjective', 'programming_language', 'pure_code']].copy()
            save_df = save_df.rename(columns={'pure_code': 'response'})
            save_df.to_parquet(output_path, index=False)
        elif format == 'json':
            # 转换为训练格式
            training_data = []
            for _, row in df.iterrows():
                training_data.append({
                    "input": row['prompt'],
                    "output": row['pure_code']
                })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)

        print(f"保存完成: {len(df):,} 条数据")


def main():
    """主函数"""

    # 配置参数
    # 创建data文件夹路径
    data_dir = "data"
    output_path = os.path.join(data_dir, "balanced_code_dataset.parquet")
    json_output_path = os.path.join(data_dir, "balanced_code_dataset.json")

    parquet_path = "part_1_200000.parquet"
    target_size = 20000  # 目标数据量

    print("开始处理代码数据集...")

    # 创建处理器
    processor = CodeDataProcessor(parquet_path)

    # 步骤1: 提取纯代码
    processed_df = processor.process_all_responses()

    # 步骤2: 质量筛选
    quality_df = processor.filter_quality_data(processed_df)

    # 步骤3: 创建平衡数据集
    balanced_df = processor.create_balanced_dataset(quality_df, target_size)

    if len(balanced_df) > 0:
        # 步骤4: 分析最终分布
        processor.analyze_final_distribution(balanced_df)

        # 步骤5: 保存数据集
        processor.save_dataset(balanced_df, output_path, format='parquet')

        # 同时保存JSON格式
        processor.save_dataset(balanced_df, json_output_path, format='json')

        print(f"\n数据处理完成！")
        print(f"原始数据: 200,000 条")
        print(f"最终数据: {len(balanced_df):,} 条")
        print(f"输出文件:")
        print(f"  - {output_path} (Parquet格式)")
        print(f"  - {json_output_path} (JSON格式)")

        # 最终统计
        final_languages = balanced_df['programming_language'].nunique()
        print(f"覆盖编程语言: {final_languages} 种")

        # 显示排除的语言信息
        excluded_languages = ["Neo4j database and Cypher", "relation database and SQL"]
        print(f"已排除的语言: {excluded_languages}")
    else:
        print("数据处理失败，未生成有效数据集")


if __name__ == "__main__":
    main()