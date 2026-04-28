from collections import Counter
from typing import Iterable

from qa_engine import normalize_text


def exact_match_score(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_ground_truth_metric(
    prediction: str, references: Iterable[str], metric_fn
) -> float:
    references = list(references)
    if not references:
        return metric_fn(prediction, "")
    return max(metric_fn(prediction, ref) for ref in references)
