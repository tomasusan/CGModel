import traceback
import torch
from typing import List, Optional, Dict
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
            tree = self.parser.parse(bytes(code, "utf8"))
            return tree
        except Exception:
            return None

    def ast_to_sequence(self, ast_tree) -> List[Dict]:
        if ast_tree is None:
            return []

        seq = []

        def traverse(node, depth=0):
            # node.type exists in tree-sitter nodes
            node_type = getattr(node, "type", None)
            if node_type is not None:
                seq.append({"type": node_type, "depth": depth})
            for c in getattr(node, "children", []) or []:
                traverse(c, depth + 1)

        traverse(ast_tree.root_node)
        return seq


def list_edit_distance(a: List[str], b: List[str]) -> int:
    """Compute classic edit distance (Levenshtein) between two lists of tokens."""
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


class BatchASTProcessor:
    def __init__(self):
        # cache processors per language
        self._processors: Dict[str, ASTProcessor] = {}

    def _get_processor(self, language: str) -> ASTProcessor:
        lang = (language or "python").lower()
        if lang not in self._processors:
            self._processors[lang] = ASTProcessor(lang)
        return self._processors[lang]

    def calculate_similarity(self, seq1: List[Dict], seq2: List[Dict]) -> float:
        if not seq1 or not seq2:
            return 0.0
        types1 = [n["type"] for n in seq1]
        types2 = [n["type"] for n in seq2]
        max_len = max(len(types1), len(types2))
        if max_len == 0:
            return 1.0
        ed = list_edit_distance(types1, types2)
        sim = 1.0 - (ed / max_len)
        return max(sim, 0.0)

    def compute_batch_ast_loss(self, teacher_codes: List[str], student_codes: List[str], languages: Optional[List[str]] = None):
        losses = []
        languages = languages or [None] * len(teacher_codes)
        for t_code, s_code, lang in zip(teacher_codes, student_codes, languages):
            try:
                proc = self._get_processor(lang)
                ast_t = proc.code_to_ast(t_code)
                ast_s = proc.code_to_ast(s_code)
                seq_t = proc.ast_to_sequence(ast_t)
                seq_s = proc.ast_to_sequence(ast_s)
                sim = self.calculate_similarity(seq_t, seq_s)
                loss = 1.0 - sim
                losses.append(loss)
            except Exception:
                traceback.print_exc()
                losses.append(1.0)

        tensor = torch.tensor(losses, dtype=torch.float32)
        return tensor
