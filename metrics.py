from typing import List

def precision_at_k(labels: List[bool], k: int) -> float:
    if not labels or k == 0:
        return 0.0
    labels_at_k = labels[:k]
    return sum(1 for x in labels_at_k if x) / float(k)

def recall_at_k(labels: List[bool], k: int) -> float:
    if not labels:
        return 0.0
    total_relevant = sum(1 for x in labels if x)
    if total_relevant == 0:
        return 0.0
    labels_at_k = labels[:k]
    return sum(1 for x in labels_at_k if x) / float(total_relevant)
