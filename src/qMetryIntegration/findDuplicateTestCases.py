"""
Find Duplicate Test Cases

A class that identifies duplicate or near-duplicate test cases based on:
- summary (exact or normalized text comparison)
- stepDetails (canonical step-by-step comparison)
- normalized_summary (exact or similarity)
- embedding (cosine similarity for semantic near-duplicates)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class DuplicateGroup:
    """Represents a group of test cases that are duplicates of each other."""

    testcases: List[Dict[str, Any]] = field(default_factory=list)
    match_reasons: List[str] = field(default_factory=list)
    similarity_score: Optional[float] = None  # For embedding-based matches

    def add(self, tc: Dict[str, Any], reason: str, score: Optional[float] = None) -> None:
        if tc not in self.testcases:
            self.testcases.append(tc)
        if reason not in self.match_reasons:
            self.match_reasons.append(reason)
        if score is not None and (self.similarity_score is None or score > self.similarity_score):
            self.similarity_score = score


class DuplicateTestCaseFinder:
    """
    Finds duplicate test cases based on summary, stepDetails, normalized_summary,
    and embedding similarity.
    """

    def __init__(
        self,
        embeddings_file: str = "qmetry_testcases_embeddings.json",
        project_root: Optional[Path] = None,
        # Thresholds
        embedding_similarity_threshold: float = 0.92,
        normalized_summary_similarity_threshold: float = 0.85,
        # Options
        normalize_text_for_comparison: bool = True,
        use_embedding_search: bool = True,
    ):
        """
        Args:
            embeddings_file: Path to embeddings JSON (relative to project root)
            project_root: Project root path. Defaults to parent of src/.
            embedding_similarity_threshold: Min cosine similarity to consider duplicates
            normalized_summary_similarity_threshold: For fuzzy normalized_summary match
            normalize_text_for_comparison: Normalize whitespace/case for text fields
            use_embedding_search: Include embedding-based duplicate detection
        """
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.embeddings_path = self.project_root / embeddings_file
        self.embedding_threshold = embedding_similarity_threshold
        self.normalized_summary_threshold = normalized_summary_similarity_threshold
        self.normalize_text = normalize_text_for_comparison
        self.use_embedding_search = use_embedding_search

        self._data: Optional[Dict[str, Any]] = None
        self._testcases: List[Dict[str, Any]] = []
        self._embeddings: Optional[List[List[float]]] = None

    def load(self) -> "DuplicateTestCaseFinder":
        """Load embeddings data from JSON file."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}\n"
                "Run createSemanticEmbeddings.py first to generate it."
            )
        with open(self.embeddings_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._testcases = self._data.get("testcases", [])
        self._embeddings = [tc.get("embedding", []) for tc in self._testcases]
        return self

    @property
    def testcases(self) -> List[Dict[str, Any]]:
        if self._testcases is None:
            self.load()
        return self._testcases

    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        """Normalize text for comparison: lowercase, collapse whitespace."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _step_details_to_canonical(step_details: Dict[str, str]) -> str:
        """Convert stepDetails dict to a canonical string for comparison."""
        if not step_details or not isinstance(step_details, dict):
            return ""
        # Sort by step number to ensure consistent ordering
        keys = sorted(
            step_details.keys(),
            key=lambda k: int(k.split("_")[1]) if "_" in k and k.split("_")[1].isdigit() else 0,
        )
        parts = [str(step_details.get(k, "")).strip() for k in keys if step_details.get(k)]
        return " || ".join(parts)

    def _get_comparison_value(self, tc: Dict[str, Any], field: str) -> str:
        """Get a string value for comparison, optionally normalized."""
        if field == "summary":
            val = tc.get("summary", "")
        elif field == "normalized_summary":
            val = tc.get("normalized_summary", "")
        elif field == "stepDetails":
            val = self._step_details_to_canonical(tc.get("stepDetails", {}))
        else:
            val = str(tc.get(field, ""))
        if self.normalize_text and isinstance(val, str):
            return self._normalize_for_comparison(val)
        return val if isinstance(val, str) else str(val)

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors (assumes normalized)."""
        if not emb1 or not emb2 or len(emb1) != len(emb2):
            return 0.0
        try:
            import numpy as np
            dot = sum(a * b for a, b in zip(emb1, emb2))
            return float(dot)  # Already normalized, dot = cosine sim
        except ImportError:
            # Fallback without numpy
            dot = sum(a * b for a, b in zip(emb1, emb2))
            return min(1.0, max(-1.0, dot))

    def find_by_summary(self) -> List[DuplicateGroup]:
        """Find duplicates where summary is identical (after normalization)."""
        groups: Dict[str, DuplicateGroup] = {}
        for tc in self.testcases:
            key = self._get_comparison_value(tc, "summary")
            if not key:
                continue
            if key not in groups:
                groups[key] = DuplicateGroup()
            groups[key].add(tc, "summary")
        return [g for g in groups.values() if len(g.testcases) > 1]

    def find_by_step_details(self) -> List[DuplicateGroup]:
        """Find duplicates where stepDetails are identical."""
        groups: Dict[str, DuplicateGroup] = {}
        for tc in self.testcases:
            key = self._get_comparison_value(tc, "stepDetails")
            if not key:
                continue
            if key not in groups:
                groups[key] = DuplicateGroup()
            groups[key].add(tc, "stepDetails")
        return [g for g in groups.values() if len(g.testcases) > 1]

    def find_by_normalized_summary(self) -> List[DuplicateGroup]:
        """Find duplicates where normalized_summary is identical."""
        groups: Dict[str, DuplicateGroup] = {}
        for tc in self.testcases:
            key = self._get_comparison_value(tc, "normalized_summary")
            if not key:
                continue
            if key not in groups:
                groups[key] = DuplicateGroup()
            groups[key].add(tc, "normalized_summary")
        return [g for g in groups.values() if len(g.testcases) > 1]

    def find_by_embedding(self) -> List[DuplicateGroup]:
        """Find near-duplicates based on embedding cosine similarity."""
        if not self.use_embedding_search or not self._embeddings:
            return []
        groups: List[DuplicateGroup] = []
        used: Set[int] = set()
        for i, tc1 in enumerate(self.testcases):
            if i in used:
                continue
            emb1 = tc1.get("embedding", [])
            if not emb1:
                continue
            group = DuplicateGroup()
            group.add(tc1, "embedding", 1.0)
            for j, tc2 in enumerate(self.testcases):
                if i == j or j in used:
                    continue
                emb2 = tc2.get("embedding", [])
                if not emb2:
                    continue
                sim = self._cosine_similarity(emb1, emb2)
                if sim >= self.embedding_threshold:
                    group.add(tc2, "embedding", sim)
                    used.add(j)
            if len(group.testcases) > 1:
                used.add(i)
                groups.append(group)
        return groups

    def find_all(
        self,
        by_summary: bool = True,
        by_step_details: bool = True,
        by_normalized_summary: bool = True,
        by_embedding: bool = True,
    ) -> Dict[str, List[DuplicateGroup]]:
        """
        Find duplicates using all configured methods.
        Returns a dict mapping method name to list of duplicate groups.
        """
        if self._testcases is None:
            self.load()
        result: Dict[str, List[DuplicateGroup]] = {}
        if by_summary:
            result["summary"] = self.find_by_summary()
        if by_step_details:
            result["stepDetails"] = self.find_by_step_details()
        if by_normalized_summary:
            result["normalized_summary"] = self.find_by_normalized_summary()
        if by_embedding and self.use_embedding_search:
            result["embedding"] = self.find_by_embedding()
        return result

    def merge_groups(self, groups: Dict[str, List[DuplicateGroup]]) -> List[DuplicateGroup]:
        """
        Merge duplicate groups that share test cases (same TC in multiple criteria).
        Returns a list of consolidated duplicate groups.
        """
        # Build a graph: tc_id -> set of tc_ids that are duplicates
        tc_to_duplicates: Dict[str, Set[str]] = {}
        for method_groups in groups.values():
            for group in method_groups:
                ids_ = [tc.get("id") or tc.get("key", str(i)) for i, tc in enumerate(group.testcases)]
                for tc_id in ids_:
                    if tc_id not in tc_to_duplicates:
                        tc_to_duplicates[tc_id] = set()
                    tc_to_duplicates[tc_id].update(ids_)

        # Union-find to merge connected components
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for tc_id, dup_ids in tc_to_duplicates.items():
            for dup_id in dup_ids:
                union(tc_id, dup_id)

        # Group by root
        roots: Dict[str, List[str]] = {}
        for tc_id in tc_to_duplicates:
            r = find(tc_id)
            if r not in roots:
                roots[r] = []
            roots[r].append(tc_id)

        # Build merged DuplicateGroups
        id_to_tc = {tc.get("id") or tc.get("key", str(i)): tc for i, tc in enumerate(self.testcases)}
        merged: List[DuplicateGroup] = []
        for ids in roots.values():
            if len(ids) < 2:
                continue
            group = DuplicateGroup()
            for tc_id in ids:
                if tc_id in id_to_tc:
                    group.add(id_to_tc[tc_id], "merged")
            merged.append(group)
        return merged

    def get_summary(
        self,
        groups: Optional[Dict[str, List[DuplicateGroup]]] = None,
        verbose: bool = True,
    ) -> str:
        """Return a human-readable summary of duplicate findings."""
        if groups is None:
            groups = self.find_all()
        lines = [
            "=" * 60,
            "Duplicate Test Cases Report",
            "=" * 60,
            f"Total test cases analyzed: {len(self.testcases)}",
            "",
        ]
        total_duplicate_tcs = 0
        for method, method_groups in groups.items():
            count = len(method_groups)
            tcs_in_groups = sum(len(g.testcases) for g in method_groups)
            total_duplicate_tcs += tcs_in_groups
            lines.append(f"By {method}: {count} duplicate group(s), {tcs_in_groups} test cases")
            if verbose:
                for i, g in enumerate(method_groups[:5]):  # Show first 5
                    keys = [tc.get("key", tc.get("id", "?")) for tc in g.testcases]
                    lines.append(f"  Group {i + 1}: {keys}")
                if count > 5:
                    lines.append(f"  ... and {count - 5} more group(s)")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find duplicate test cases based on summary, stepDetails, normalized_summary, and embedding similarity"
    )
    parser.add_argument(
        "--file",
        "-f",
        default="qmetry_testcases_embeddings.json",
        help="Path to embeddings JSON file",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Disable embedding-based duplicate detection",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.92,
        help="Embedding similarity threshold (default: 0.92)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge duplicate groups that share test cases",
    )
    args = parser.parse_args()

    finder = DuplicateTestCaseFinder(
        embeddings_file=args.file,
        embedding_similarity_threshold=args.threshold,
        use_embedding_search=not args.no_embedding,
    )
    finder.load()

    groups = finder.find_all()
    if args.merge:
        merged = finder.merge_groups(groups)
        print(f"\nMerged duplicate groups: {len(merged)}")
        for i, g in enumerate(merged):
            keys = [tc.get("key", tc.get("id", "?")) for tc in g.testcases]
            print(f"  Group {i + 1}: {keys}")
    else:
        print(finder.get_summary(groups))
