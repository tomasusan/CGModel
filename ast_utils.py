import hashlib
import traceback
import torch
from typing import List, Optional, Dict, Set
from tree_sitter_languages import get_parser


class ASTProcessor:
    def __init__(self, language: str = "python"):
        self.language = language
        try:
            self.parser = get_parser(language)
        except Exception:
            self.parser = None

    def code_to_ast(self, code: str):
        if self.parser is None:
            return None
        try:
            return self.parser.parse(bytes(code, "utf8"))
        except Exception:
            return None

    # ================================
    # 🔥 Subtree Hash Extraction
    # ================================
    def extract_subtree_hashes(self, ast_tree) -> Set[str]:
        if ast_tree is None:
            return set()

        subtree_hashes = set()

        def hash_node(node):
            if node is None:
                return ""

            child_hashes = []
            for child in node.children:
                child_hashes.append(hash_node(child))

            # 结构签名 = type + ordered children hash
            signature = node.type + "(" + ",".join(child_hashes) + ")"

            h = hashlib.md5(signature.encode()).hexdigest()

            subtree_hashes.add(h)
            return h

        hash_node(ast_tree.root_node)
        return subtree_hashes

class BatchASTProcessor:
    def __init__(self):
        self._processors: Dict[str, ASTProcessor] = {}

    def _get_processor(self, language: str) -> ASTProcessor:
        lang = (language or "python").lower()
        if lang not in self._processors:
            self._processors[lang] = ASTProcessor(lang)
        return self._processors[lang]

    # ==========================================
    # Tree Kernel Similarity
    # ==========================================
    def calculate_similarity(self, hashes1: Set[str], hashes2: Set[str]) -> float:
        if not hashes1 or not hashes2:
            return 0.0

        intersection = len(hashes1.intersection(hashes2))
        union = len(hashes1.union(hashes2))

        if union == 0:
            return 1.0

        return intersection / union

    # ==========================================
    # AST Loss
    # =========================================
    def compute_batch_ast_loss(
        self,
        teacher_codes: List[str],
        student_codes: List[str],
        languages: Optional[List[str]] = None
    ):
        losses = []
        languages = languages or [None] * len(teacher_codes)

        for t_code, s_code, lang in zip(teacher_codes, student_codes, languages):
            try:
                proc = self._get_processor(lang)

                ast_t = proc.code_to_ast(t_code)
                ast_s = proc.code_to_ast(s_code)

                hashes_t = proc.extract_subtree_hashes(ast_t)
                hashes_s = proc.extract_subtree_hashes(ast_s)

                sim = self.calculate_similarity(hashes_t, hashes_s)

                loss = 1.0 - sim
                losses.append(loss)

            except Exception:
                traceback.print_exc()
                losses.append(1.0)

        return torch.tensor(losses, dtype=torch.float32)
