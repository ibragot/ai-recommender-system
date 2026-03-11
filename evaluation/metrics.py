import numpy as np
from typing import List


def rmse(actual: List[float], predicted: List[float]) -> float:
    """Root Mean Squared Error. Lower is better. Perfect = 0."""
    a, p = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual: List[float], predicted: List[float]) -> float:
    """Mean Absolute Error. Lower is better."""
    a, p = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(a - p)))


def precision_at_k(recommended, relevant, k=10) -> float:
    """Of the top-K recommendations, what % were actually good?"""
    if not recommended or k == 0:
        return 0.0
    hits = sum(1 for m in recommended[:k] if m in set(relevant))
    return hits / k


def recall_at_k(recommended, relevant, k=10) -> float:
    """Of all good movies, what % did we find in top-K?"""
    if not relevant or k == 0:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10) -> float:
    """NDCG: rewards putting the BEST picks at the very top. 1.0 = perfect."""
    if not recommended or not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(recommended[:k]) if m in relevant_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0
