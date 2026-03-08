import re
from typing import Dict, List, Set, Tuple
from .embedding_matcher import EmbeddingMatcher


class Evaluator:
    """
    Оценщик, который:
    1) сначала сопоставляет предметы по словарю синонимов;
    2) затем для оставшихся использует семантическое сходство эмбеддингов.
    """

    def __init__(self, descriptions: Dict[int, List[str]]):
        self.synonym_map = {synonyms[0]: synonyms[1:] for synonyms in descriptions.values()}

        self.reverse_map: Dict[str, str] = {}
        for canonical, synonyms in self.synonym_map.items():
            canonical_norm = canonical.strip().lower()
            self.reverse_map[canonical_norm] = canonical_norm
            for syn in synonyms:
                syn_norm = syn.strip().lower()
                self.reverse_map[syn_norm] = canonical_norm

        self.embedding_matcher = EmbeddingMatcher()

    def _parse_items(self, text: str) -> List[str]:
        if text.strip().lower() == "none":
            return []

        items = []
        for match in re.findall(r"\[([^\]]+)\]", text):
            for item in match.split(","):
                item = item.strip().lower()
                if item:
                    items.append(item)
        return items

    def _canonical(self, item: str) -> str:
        key = item.strip().lower()
        return self.reverse_map.get(key, key)

    def _greedy_matching(
        self, pred_items: List[str], gt_items: List[str]
    ) -> Tuple[List[Tuple[str, str]], Set[str], Set[str]]:
        """
        1) Матчит по словарю синонимов.
        2) Оставшиеся — по эмбеддингам через EmbeddingMatcher.
        """
        num_pred = len(pred_items)
        num_gt = len(gt_items)

        matched_pred_idx: Set[int] = set()
        matched_gt_idx: Set[int] = set()
        matched_pairs: List[Tuple[str, str]] = []

        # 1) Сопоставление по словарю синонимов
        for p_idx, p_item in enumerate(pred_items):
            canon_p = self._canonical(p_item)
            for g_idx, g_item in enumerate(gt_items):
                if g_idx in matched_gt_idx:
                    continue
                canon_g = self._canonical(g_item)
                if canon_p == canon_g:
                    matched_pred_idx.add(p_idx)
                    matched_gt_idx.add(g_idx)
                    matched_pairs.append((p_item, g_item))
                    break

        # 2) Сопоставление по эмбеддингам для оставшихся
        remaining_pred_indices = [i for i in range(num_pred) if i not in matched_pred_idx]
        remaining_gt_indices = [j for j in range(num_gt) if j not in matched_gt_idx]

        if remaining_pred_indices and remaining_gt_indices:
            remaining_pred_items = [pred_items[i] for i in remaining_pred_indices]
            remaining_gt_items = [gt_items[j] for j in remaining_gt_indices]

            emb_pred_local, emb_gt_local = self.embedding_matcher.find_best_matches(
                remaining_pred_items, remaining_gt_items
            )

            for local_p, local_g in zip(emb_pred_local, emb_gt_local):
                p_idx = remaining_pred_indices[local_p]
                g_idx = remaining_gt_indices[local_g]

                if p_idx in matched_pred_idx or g_idx in matched_gt_idx:
                    continue

                matched_pred_idx.add(p_idx)
                matched_gt_idx.add(g_idx)
                matched_pairs.append((pred_items[p_idx], gt_items[g_idx]))

        fp = {item for i, item in enumerate(pred_items) if i not in matched_pred_idx}
        fn = {item for j, item in enumerate(gt_items) if j not in matched_gt_idx}

        return matched_pairs, fp, fn

    def evaluate_pair(self, prediction: str, ground_truth: str) -> Dict:
        pred_items = self._parse_items(prediction)
        gt_items = self._parse_items(ground_truth)

        tp_pairs, fp, fn = self._greedy_matching(pred_items, gt_items)

        tp = [f"{p} - {g}" for p, g in tp_pairs]

        return {
            "tp": sorted(tp),
            "fp": sorted(fp),
            "fn": sorted(fn),
        }

    def evaluate(self, pred_data: Dict[str, str], gt_data: Dict[str, str]) -> List[Dict]:
        results = []

        common_ids = set(pred_data.keys()) & set(gt_data.keys())

        for id_key in sorted(common_ids):
            result = self.evaluate_pair(pred_data[id_key], gt_data[id_key])
            result["id"] = id_key
            results.append(result)

        return results
