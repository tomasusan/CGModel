import re
import pandas as pd

# =========================
# Language Mapping Dictionary
# =========================
# Maps programming language names from the dataset to tree-sitter compatible identifiers
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
# Code Cleaning Function
# =========================
def clean_response_text(response):
    """
    Extract pure code from markdown-style response text.

    This function uses regex to extract code blocks delimited by triple backticks.
    If multiple code blocks are found, they are joined with newlines.

    Args:
        response: Raw response text that may contain code blocks

    Returns:
        Extracted code string, or empty string if no valid code found
    """
    if pd.isna(response) or not isinstance(response, str):
        return ""

    # Pattern to match code blocks: ```language\ncode```
    pattern = r'```[\w]*\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        # Join multiple code blocks with newlines
        return '\n'.join(matches).strip()
    return ""

from ast_utils import ASTProcessor

# =========================
# AST Validation Function
# =========================
def filter_valid_ast_rows(df):
    """
    Filter dataset rows based on AST parsing success.

    This function attempts to parse each code sample into an Abstract Syntax Tree
    using tree-sitter. Only samples that parse successfully without syntax errors
    are kept in the dataset.

    Args:
        df: Input DataFrame with 'response' and 'programming_language' columns

    Returns:
        DataFrame containing only rows that passed AST validation
    """
    print("Performing AST validation...")

    # Cache for AST processors to avoid recreating for same language
    processors = {}
    valid_mask = []

    # Iterate through each row to validate AST parsing
    for idx, row in df.iterrows():
        code = row["response"]
        raw_lang = row["programming_language"]

        # Map raw language name to tree-sitter compatible identifier
        mapped_lang = language_type_map.get(raw_lang)

        # Skip if language is not supported
        if mapped_lang is None:
            valid_mask.append(False)
            continue

        # Get or create cached processor for this language
        if mapped_lang not in processors:
            processors[mapped_lang] = ASTProcessor(mapped_lang)

        processor = processors[mapped_lang]

        # Skip if parser initialization failed
        if processor.parser is None:
            valid_mask.append(False)
            continue

        try:
            # Attempt to parse the code
            tree = processor.code_to_ast(code)

            # Check parsing results
            if tree is None:
                valid_mask.append(False)
            elif tree.root_node.has_error:
                # Tree-sitter reports syntax errors in the AST
                valid_mask.append(False)
            else:
                # Successfully parsed without errors
                valid_mask.append(True)

        except Exception:
            # Any exception during parsing results in failure
            valid_mask.append(False)

    # Add validation results as a new column
    df["ast_valid"] = valid_mask
    # Keep only valid rows and drop the temporary column
    df_valid = df[df["ast_valid"]].drop(columns=["ast_valid"])

    # Print validation statistics
    print(f"Before AST filtering: {len(df)}")
    print(f"After AST filtering: {len(df_valid)}")
    print(f"AST success rate: {len(df_valid) / len(df):.4f}")

    return df_valid


# =========================
# Main Processing Function
# =========================
def process_parquet_data(file_path, output_path, sample_size=10000):
    """
    Main pipeline for processing raw parquet data into a clean, balanced dataset.

    Processing steps:
    1. Read parquet file
    2. Filter to required columns
    3. Exclude unwanted programming languages
    4. Clean response text (extract code)
    5. Validate AST parsing
    6. Balance dataset by language and difficulty
    7. Sample to target size
    8. Save processed dataset

    Args:
        file_path: Path to input parquet file
        output_path: Path where processed dataset will be saved
        sample_size: Target number of samples in final dataset

    Returns:
        Processed DataFrame
    """
    print("Reading data...")
    df = pd.read_parquet(file_path)

    print("Filtering columns...")
    # Define target columns for the final dataset
    target_columns = ['prompt', 'adjective', 'programming_language', 'response']
    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in target_columns if col in df.columns]
    df_filtered = df[existing_columns].copy()

    print("Filtering programming languages...")
    # Define languages to exclude from the dataset
    exclude_languages = ['Neo4j database and Cypher', 'relation database and SQL']
    df_filtered = df_filtered[~df_filtered['programming_language'].isin(exclude_languages)]

    print("Cleaning response column...")
    # Extract pure code from responses
    df_filtered['response'] = df_filtered['response'].apply(clean_response_text)
    # Remove rows with empty responses after cleaning
    df_filtered = df_filtered[df_filtered['response'] != ""]

    # =========================
    # AST Validation Step
    # =========================
    df_filtered = filter_valid_ast_rows(df_filtered)

    # Verify difficulty column exists
    if 'adjective' not in df_filtered.columns:
        raise ValueError("Difficulty column (adjective) not found in data")

    # Target proportions for difficulty levels (approximately balanced)
    adjective_proportions = {
        'Low': 0.336,
        'High': 0.334,
        'Extreme': 0.330
    }

    print("Sampling data proportionally...")
    sampled_dfs = []
    languages = df_filtered['programming_language'].unique()

    # Calculate base samples per language
    lang_sample_base = sample_size / len(languages)

    # Sample from each language
    for lang in languages:
        lang_df = df_filtered[df_filtered['programming_language'] == lang].copy()
        lang_sampled = []

        # Sample from each difficulty level according to proportions
        for adjective, proportion in adjective_proportions.items():
            diff_df = lang_df[lang_df['adjective'] == adjective].copy()
            diff_sample_size = int(round(lang_sample_base * proportion))

            # Sample or take all if insufficient data
            if len(diff_df) <= diff_sample_size:
                sampled = diff_df
            else:
                sampled = diff_df.sample(n=diff_sample_size, random_state=42)

            lang_sampled.append(sampled)

        # Combine difficulty samples for this language
        sampled_dfs.append(pd.concat(lang_sampled, ignore_index=True))

    # Combine all language samples
    final_df = pd.concat(sampled_dfs, ignore_index=True)

    # If we overshot the target size, randomly sample down
    if len(final_df) > sample_size:
        final_df = final_df.sample(n=sample_size, random_state=42)

    print(f"Final dataset size: {len(final_df)}")

    # Save processed dataset
    final_df.to_parquet(output_path, index=False)

    print("Data processing complete!")

    return final_df


# Main execution block
if __name__ == "__main__":
    # Define input and output file paths
    input_file = "part_1_200000.parquet"
    output_file = "data/Output_data.parquet"

    # Run the data processing pipeline
    process_parquet_data(input_file, output_file, sample_size=10000)