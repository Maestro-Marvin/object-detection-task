from typing import Dict, List


def calculate_metrics(evaluation_results: List[Dict]) -> Dict:
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