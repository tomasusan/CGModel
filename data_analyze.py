import pandas as pd
from collections import defaultdict

from ast_utils import ASTProcessor

# Mapping dictionary for converting programming language names from the dataset
# to tree-sitter compatible language identifiers
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
    """
    Validate a dataset of code samples by attempting to parse them into ASTs.

    This function reads a parquet file containing code samples with associated
    programming languages, attempts to parse each sample using tree-sitter,
    and generates statistics on parsing success rates. Failed samples are
    saved to a separate file with failure reasons.

    Args:
        file_path: Path to the input parquet file containing the dataset
        failed_output_path: Path where failed samples will be saved
    """
    # Read the dataset from parquet file
    df = pd.read_parquet(file_path)

    # Get total number of rows for statistics
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    print("=" * 70)

    # Initialize counters for success and failure tracking
    success_count = 0
    fail_count = 0

    # Dictionary to track statistics per programming language
    lang_stats = defaultdict(lambda: {"success": 0, "fail": 0})

    # List to store failed samples with failure reasons
    failed_samples = []

    # Cache for AST processors to avoid recreating for the same language
    processors = {}

    # Iterate through each row in the dataset
    for idx, row in df.iterrows():
        # Extract code and language from the current row
        code = row.get("response", None)
        raw_lang = row.get("programming_language", None)

        # Variable to store the reason for failure (if any)
        fail_reason = None

        # 1️⃣ Check if code is present and non-empty
        if not code or not str(code).strip():
            fail_reason = "missing_code"

        # 2️⃣ Check if language field is present
        elif not raw_lang:
            fail_reason = "missing_language"

        else:
            # Map the raw language name to tree-sitter compatible format
            mapped_lang = language_type_map.get(raw_lang, None)

            # 3️⃣ Check if language is supported by tree-sitter
            if mapped_lang is None:
                fail_reason = "unsupported_language"

            else:
                # 4️⃣ Get or create AST processor for this language
                if mapped_lang not in processors:
                    processors[mapped_lang] = ASTProcessor(mapped_lang)

                processor = processors[mapped_lang]

                # Check if parser was successfully initialized
                if processor.parser is None:
                    fail_reason = "parser_init_failed"
                else:
                    try:
                        # Attempt to parse the code into an AST
                        tree = processor.code_to_ast(code)

                        # Check parsing results
                        if tree is None:
                            fail_reason = "parse_exception"
                        elif tree.root_node.has_error:
                            # Tree-sitter reports syntax errors in the AST
                            fail_reason = "syntax_error"
                        else:
                            # Successfully parsed without errors
                            success_count += 1
                            lang_stats[raw_lang]["success"] += 1
                            continue  # Skip failure handling for successful samples

                    except Exception:
                        # Catch any unexpected exceptions during parsing
                        fail_reason = "parse_exception"

        # If we reach here, the current sample failed validation
        fail_count += 1
        lang_stats[str(raw_lang)]["fail"] += 1

        # Create failed sample record with failure reason
        failed_sample = row.to_dict()
        failed_sample["fail_reason"] = fail_reason
        failed_samples.append(failed_sample)

    # Output overall statistics
    print(f"AST Parse Success: {success_count}")
    print(f"AST Parse Fail   : {fail_count}")
    print(f"Success Rate     : {success_count / total_rows:.4f}")

    # Output per-language statistics
    print("\nPer Language Statistics:")
    for lang, stats in lang_stats.items():
        total = stats["success"] + stats["fail"]
        rate = stats["success"] / total if total > 0 else 0
        print(
            f"{lang:12} | total: {total:6} | success: {stats['success']:6} | "
            f"fail: {stats['fail']:6} | success_rate: {rate:.4f}"
        )

    # Save failed samples to output file if any exist
    if failed_samples:
        failed_df = pd.DataFrame(failed_samples)
        failed_df.to_parquet(failed_output_path, index=False)
        print(f"\nFailed samples saved to: {failed_output_path}")
        print(f"Total failed samples: {len(failed_samples)}")
    else:
        print("\nNo failed samples.")


# Main execution block
if __name__ == "__main__":
    # Define input and output file paths
    file_path = "data/Output_data.parquet"
    failed_output_path = "data/failed_samples.parquet"

    # Run dataset validation
    validate_dataset(file_path, failed_output_path)