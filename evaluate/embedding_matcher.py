from typing import List, Tuple, Optional
import torch
from vllm import LLM
from config import EMBED_MODEL_NAME, SIMILARITY_THRESHOLD

class EmbeddingMatcher:

    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.llm = LLM(
            model=EMBED_MODEL_NAME,
            runner="pooling",
            gpu_memory_utilization=0.15,
            trust_remote_code=True,
            max_model_len=32768
        )
        self.task_instruction = (
            "Given a household object name, retrieve semantically similar objects from a scene."
        )

    def _format_query(self, item: str) -> str:
        return f"Instruct: {self.task_instruction}\nQuery: {item}"

    def compute_similarities(self, pred_items: List[str], gt_items: List[str]) -> torch.Tensor:
        if not pred_items or not gt_items:
            return torch.zeros(len(pred_items), len(gt_items))

        queries = [self._format_query(item) for item in pred_items]
        documents = gt_items

        input_texts = queries + documents

        outputs = self.llm.embed(input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs], dtype=torch.float32)

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        pred_embeddings = embeddings[:len(queries)]
        gt_embeddings = embeddings[len(queries):]

        similarity_matrix = pred_embeddings @ gt_embeddings.T
        return similarity_matrix

    def find_best_matches(
        self,
        pred_items: List[str],
        gt_items: List[str],
    ) -> Tuple[List[int], List[int]]:
        if not pred_items or not gt_items:
            return [], []

        sim_matrix = self.compute_similarities(pred_items, gt_items)

        matched_pred = set()
        matched_gt = set()
        matches: List[Tuple[int, int]] = []

        while True:
            max_val = sim_matrix.max().item()
            if max_val < self.similarity_threshold:
                break

            flat_idx = sim_matrix.argmax().item()
            pred_idx, gt_idx = divmod(flat_idx, sim_matrix.shape[1])

            if pred_idx in matched_pred or gt_idx in matched_gt:
                sim_matrix[pred_idx, gt_idx] = -1.0
                continue

            matches.append((pred_idx, gt_idx))
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

            sim_matrix[pred_idx, :] = -1.0
            sim_matrix[:, gt_idx] = -1.0

        pred_indices = [p for p, _ in matches]
        gt_indices = [g for _, g in matches]
        return pred_indices, gt_indices