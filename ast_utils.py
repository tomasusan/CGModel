import hashlib
import traceback
import torch
from typing import List, Optional, Dict, Set
from tree_sitter_languages import get_parser


class ASTProcessor:
    """
    Abstract Syntax Tree (AST) Processor for code analysis.

    This class handles parsing source code into ASTs and extracting structural
    features such as subtree hashes for similarity comparison between code snippets.
    """

    def __init__(self, language: str = "python"):
        """
        Initialize the AST processor for a specific programming language.

        Args:
            language: Programming language name (e.g., "python", "java", "cpp")
        """
        self.language = language
        try:
            # Attempt to get the tree-sitter parser for the specified language
            self.parser = get_parser(language)
        except Exception:
            # Fallback if parser is unavailable for the language
            self.parser = None

    def code_to_ast(self, code: str):
        """
        Parse source code into an Abstract Syntax Tree.

        Args:
            code: Source code string to parse

        Returns:
            AST tree object if parsing successful, None otherwise
        """
        if self.parser is None:
            return None
        try:
            # Parse the code bytes into an AST
            return self.parser.parse(bytes(code, "utf8"))
        except Exception:
            # Return None if parsing fails (e.g., syntax errors)
            return None

    # ================================
    # Subtree Hash Extraction
    # ================================
    def extract_subtree_hashes(self, ast_tree) -> Set[str]:
        """
        Extract unique hash signatures for all subtrees in the AST.

        This method traverses the AST and generates MD5 hashes for each subtree
        based on node types and child structures. These hashes can be used for
        structural similarity comparison between different ASTs.

        Args:
            ast_tree: AST tree object from code_to_ast()

        Returns:
            Set of unique hash strings representing all subtrees in the AST
        """
        if ast_tree is None:
            return set()

        subtree_hashes = set()

        def hash_node(node):
            """
            Recursively hash a node and all its children.

            Args:
                node: Current AST node

            Returns:
                Hash string for the current node
            """
            if node is None:
                return ""

            # Recursively compute hashes for all child nodes
            child_hashes = []
            for child in node.children:
                child_hashes.append(hash_node(child))

            # Create structural signature: node type + ordered list of child hashes
            signature = node.type + "(" + ",".join(child_hashes) + ")"

            # Generate MD5 hash of the signature
            h = hashlib.md5(signature.encode()).hexdigest()

            # Add hash to the set of all subtree hashes
            subtree_hashes.add(h)
            return h

        # Start recursive hashing from the root node
        hash_node(ast_tree.root_node)
        return subtree_hashes


class BatchASTProcessor:
    """
    Batch processor for AST-based operations across multiple code samples.

    This class manages AST processors for different languages and provides
    methods for computing structural similarities and losses between batches
    of teacher and student code generations.
    """

    def __init__(self):
        # Cache of AST processors for different languages
        self._processors: Dict[str, ASTProcessor] = {}

    def _get_processor(self, language: str) -> ASTProcessor:
        """
        Get or create an AST processor for the specified language.

        Args:
            language: Programming language name

        Returns:
            ASTProcessor instance for the language
        """
        lang = (language or "python").lower()
        if lang not in self._processors:
            # Create new processor if not already cached
            self._processors[lang] = ASTProcessor(lang)
        return self._processors[lang]

    # ==========================================
    # Tree Kernel Similarity Computation
    # ==========================================
    def calculate_similarity(self, hashes1: Set[str], hashes2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets of subtree hashes.

        This implements a tree kernel approximation by comparing the sets
        of structural fingerprints extracted from two ASTs.

        Args:
            hashes1: Set of subtree hashes from first AST
            hashes2: Set of subtree hashes from second AST

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        if not hashes1 or not hashes2:
            return 0.0

        # Compute Jaccard similarity: intersection size / union size
        intersection = len(hashes1.intersection(hashes2))
        union = len(hashes1.union(hashes2))

        # Avoid division by zero
        if union == 0:
            return 1.0

        return intersection / union

    # ==========================================
    # Batch AST Loss Computation
    # =========================================
    def compute_batch_ast_loss(
            self,
            teacher_codes: List[str],
            student_codes: List[str],
            languages: Optional[List[str]] = None
    ):
        """
        Compute AST-based structural losses for a batch of code pairs.

        For each pair of teacher and student code, this method:
        1. Parses both code snippets into ASTs
        2. Extracts subtree hash fingerprints
        3. Computes structural similarity between the ASTs
        4. Converts similarity to loss (1 - similarity)

        Args:
            teacher_codes: List of reference code strings from teacher model
            student_codes: List of generated code strings from student model
            languages: Optional list of programming languages for each pair
                      (defaults to Python for all)

        Returns:
            Tensor of loss values for each sample in the batch
        """
        losses = []
        # Default to None for all languages if not provided
        languages = languages or [None] * len(teacher_codes)

        # Process each code pair in the batch
        for t_code, s_code, lang in zip(teacher_codes, student_codes, languages):
            try:
                # Get appropriate processor for the language
                proc = self._get_processor(lang)

                # Parse both code snippets into ASTs
                ast_t = proc.code_to_ast(t_code)
                ast_s = proc.code_to_ast(s_code)

                # Extract structural fingerprints
                hashes_t = proc.extract_subtree_hashes(ast_t)
                hashes_s = proc.extract_subtree_hashes(ast_s)

                # Compute structural similarity
                sim = self.calculate_similarity(hashes_t, hashes_s)

                # Convert similarity to loss (higher similarity = lower loss)
                loss = 1.0 - sim
                losses.append(loss)

            except Exception:
                # Print error traceback for debugging
                traceback.print_exc()
                # Assign maximum loss (1.0) if computation fails
                losses.append(1.0)

        # Return losses as a PyTorch tensor
        return torch.tensor(losses, dtype=torch.float32)
