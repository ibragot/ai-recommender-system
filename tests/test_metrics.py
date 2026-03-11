"""
Tests for our evaluation metrics.
These run automatically every time we push to GitHub.
"""

from evaluation.metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k


# ── RMSE Tests ────────────────────────────────────────────────────────

def test_rmse_perfect():
    """If predictions exactly match actuals, RMSE should be 0."""
    actual = [4.0, 3.0, 5.0, 2.0]
    predicted = [4.0, 3.0, 5.0, 2.0]
    assert rmse(actual, predicted) == 0.0

def test_rmse_known_value():
    """Test RMSE with a known result we can verify by hand."""
    actual = [4.0, 3.0]
    predicted = [3.0, 4.0]
    # Errors are -1 and +1, squared = 1 and 1, mean = 1.0, sqrt = 1.0
    assert abs(rmse(actual, predicted) - 1.0) < 0.0001

def test_rmse_always_positive():
    """RMSE can never be negative."""
    actual = [5.0, 1.0, 3.0]
    predicted = [1.0, 5.0, 3.0]
    assert rmse(actual, predicted) >= 0.0


# ── MAE Tests ────────────────────────────────────────────────────────

def test_mae_perfect():
    """Perfect predictions should give MAE of 0."""
    actual = [4.0, 3.0, 5.0]
    predicted = [4.0, 3.0, 5.0]
    assert mae(actual, predicted) == 0.0

def test_mae_known_value():
    """Test MAE with known values."""
    actual = [4.0, 3.0, 5.0]
    predicted = [3.0, 3.0, 5.0]
    # Only first is wrong by 1.0, others are perfect
    # MAE = (1.0 + 0.0 + 0.0) / 3 = 0.333
    assert abs(mae(actual, predicted) - 0.3333) < 0.001


# ── Precision@K Tests ────────────────────────────────────────────────

def test_precision_at_k_perfect():
    """All recommendations are relevant — precision should be 1.0."""
    recommended = [1, 2, 3, 4, 5]
    relevant = [1, 2, 3, 4, 5]
    assert precision_at_k(recommended, relevant, k=5) == 1.0

def test_precision_at_k_none():
    """No recommendations are relevant — precision should be 0.0."""
    recommended = [1, 2, 3]
    relevant = [4, 5, 6]
    assert precision_at_k(recommended, relevant, k=3) == 0.0

def test_precision_at_k_partial():
    """Half the recommendations are relevant."""
    recommended = [1, 2, 3, 4]
    relevant = [1, 2]
    # 2 out of 4 are relevant = 0.5
    assert precision_at_k(recommended, relevant, k=4) == 0.5


# ── Recall@K Tests ────────────────────────────────────────────────────

def test_recall_at_k_perfect():
    """All relevant items are recommended."""
    recommended = [1, 2, 3]
    relevant = [1, 2, 3]
    assert recall_at_k(recommended, relevant, k=3) == 1.0

def test_recall_at_k_partial():
    """We found 2 out of 4 relevant items."""
    recommended = [1, 2, 5, 6]
    relevant = [1, 2, 3, 4]
    # Found 1 and 2, missed 3 and 4 = 2/4 = 0.5
    assert recall_at_k(recommended, relevant, k=4) == 0.5


# ── NDCG@K Tests ──────────────────────────────────────────────────────

def test_ndcg_perfect():
    """Perfect ranking should give NDCG of 1.0."""
    recommended = [1, 2, 3]
    relevant = [1, 2, 3]
    assert abs(ndcg_at_k(recommended, relevant, k=3) - 1.0) < 0.0001

def test_ndcg_empty():
    """Empty inputs should return 0.0."""
    assert ndcg_at_k([], [1, 2, 3], k=3) == 0.0
    assert ndcg_at_k([1, 2, 3], [], k=3) == 0.0
