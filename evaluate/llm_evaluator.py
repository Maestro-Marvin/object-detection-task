import re
import json
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from config import LLM_MODEL_NAME

class LlmEvaluator:
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.1,
            max_model_len=8192 
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1024
        )


    def _create_prompt(self, id_key: str, prediction: str, ground_truth: str) -> str:
        prompt = f"""
        You are an expert evaluator for spatial scene understanding.

        Task: Compare PREDICTION and GROUND TRUTH for object ID: "{id_key}".
        Both inputs describe small objects with their spatial relations to a reference object.

        Format of each statement: "[item1, item2] relation reference_object id=XXX"

        Rules:
        1. Parse each statement into (items, relation, container).
        2. Match items semantically: "bowl" ≈ "blue bowl", "socket" ≈ "plug socket".
        3. Relation must match exactly: "on" ≠ "inside" ≠ "near".
        4. If GT is "none" but prediction has items → all FP.
        5. If prediction is "none" but GT has items → all FN.

        PREDICTION: {prediction}
        GROUND TRUTH: {ground_truth}

        Output JSON with:
        - "tp": list of correctly predicted items (use GT wording),
        - "fp": list of falsely predicted items,
        - "fn": list of missed GT items,
        - "explanation": short justification.

        Return ONLY valid JSON. No extra text.

        Example:
        Input: 
        PREDICTION: "[lamp] on desk id=42"
        GROUND TRUTH: "[desk lamp] on desk id=42"
        Output:
        {{"tp": ["desk lamp"], "fp": [], "fn": [], "explanation": "Semantic match"}}
        """.strip()
        return prompt

    def evaluate_batch(self, gt_data: Dict[str, str], pred_data: Dict[str, str]) -> List[Dict[str, Any]]:
        common_ids = set(gt_data.keys()) & set(pred_data.keys())
        results = []

        # Handle missing/extra IDs
        for id_key in set(gt_data.keys()) - common_ids:
            results.append({
                "id": id_key,
                "tp": [], "fp": [], "fn": self._extract_items(gt_data[id_key]),
                "explanation": "Missing in prediction"
            })
        for id_key in set(pred_data.keys()) - common_ids:
            results.append({
                "id": id_key,
                "tp": [], "fp": self._extract_items(pred_data[id_key]), "fn": [],
                "explanation": "Extra in prediction"
            })

        # Process common IDs with LLM
        prompts = []
        id_list = []
        for id_key in sorted(common_ids):  # deterministic order
            prompt = self._create_prompt(id_key, pred_data[id_key], gt_data[id_key])
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)
            id_list.append(id_key)

        if prompts:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
            for i, output in enumerate(outputs):
                response = output.outputs[0].text.strip()
                parsed = self._parse_json_response(response)
                parsed["id"] = id_list[i]
                results.append(parsed)

        return results

    def _extract_items(self, text: str) -> List[str]:
        """Extract all items from raw string (fallback for missing/extra)."""
        if text.strip().lower() == "none":
            return []
        items = []
        for match in re.findall(r'\[([^\]]+)\]', text):
            items.extend([x.strip() for x in match.split(",") if x.strip()])
        return items

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            # Normalize keys to lowercase
            return {
                "tp": data.get("tp", data.get("TP", [])),
                "fp": data.get("fp", data.get("FP", [])),
                "fn": data.get("fn", data.get("FN", [])),
                "explanation": data.get("explanation", data.get("Explanation", "Parse error"))
            }
        except:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    data = json.loads(response[start:end])
                    return {
                        "tp": data.get("tp", []),
                        "fp": data.get("fp", []),
                        "fn": data.get("fn", []),
                        "explanation": data.get("explanation", "Fallback parse")
                    }
                except:
                    pass
        return {"tp": [], "fp": [], "fn": [], "explanation": "Failed to parse"}

    def calculate_metrics(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        total_tp = sum(len(r["tp"]) for r in evaluation_results)
        total_fp = sum(len(r["fp"]) for r in evaluation_results)
        total_fn = sum(len(r["fn"]) for r in evaluation_results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Total_TP": total_tp,
            "Total_FP": total_fp,
            "Total_FN": total_fn
        }