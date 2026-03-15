from typing import Dict, List, Tuple


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def calculate_metrics(evaluation_results: List[Dict]) -> Dict:
    # Микро: один набор метрик по суммарным TP/FP/FN по всем сэмплам
    total_tp = sum(len(r["tp"]) for r in evaluation_results)
    total_fp = sum(len(r["fp"]) for r in evaluation_results)
    total_fn = sum(len(r["fn"]) for r in evaluation_results)

    micro_p, micro_r, micro_f1 = _precision_recall_f1(total_tp, total_fp, total_fn)

    # Макро: считаем P/R/F1 по каждому сэмплу, затем усредняем
    macro_p_list = []
    macro_r_list = []
    macro_f1_list = []
    for r in evaluation_results:
        tp, fp, fn = len(r["tp"]), len(r["fp"]), len(r["fn"])
        p, rec, f1 = _precision_recall_f1(tp, fp, fn)
        macro_p_list.append(p)
        macro_r_list.append(rec)
        macro_f1_list.append(f1)

    n = len(evaluation_results)
    macro_p = sum(macro_p_list) / n if n > 0 else 0.0
    macro_r = sum(macro_r_list) / n if n > 0 else 0.0
    macro_f1 = sum(macro_f1_list) / n if n > 0 else 0.0

    return {
        "Micro_Precision": round(micro_p, 4),
        "Micro_Recall": round(micro_r, 4),
        "Micro_F1": round(micro_f1, 4),
        "Macro_Precision": round(macro_p, 4),
        "Macro_Recall": round(macro_r, 4),
        "Macro_F1": round(macro_f1, 4),
    }