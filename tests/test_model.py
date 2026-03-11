"""
Tests for the PyTorch recommendation model.
These check the model structure works correctly
without needing trained weights or a database.
"""

import torch
from model.recommender import MatrixFactorization


def test_model_output_shape():
    """Model should return one prediction per input pair."""
    model = MatrixFactorization(num_users=10, num_movies=20, embedding_dim=8)
    user_ids = torch.LongTensor([0, 1, 2])
    movie_ids = torch.LongTensor([0, 1, 2])
    output = model(user_ids, movie_ids)
    # Should return 3 predictions for 3 input pairs
    assert output.shape == (3,)


def test_model_rating_range():
    """All predictions should be clamped between 0.5 and 5.0."""
    model = MatrixFactorization(num_users=50, num_movies=100, embedding_dim=16)
    user_ids = torch.LongTensor(list(range(50)))
    movie_ids = torch.LongTensor(list(range(50)))
    output = model.predict(user_ids, movie_ids)
    assert output.min().item() >= 0.5
    assert output.max().item() <= 5.0


def test_model_single_prediction():
    """Model should handle a single user-movie pair."""
    model = MatrixFactorization(num_users=5, num_movies=10, embedding_dim=4)
    user_id = torch.LongTensor([0])
    movie_id = torch.LongTensor([0])
    output = model.predict(user_id, movie_id)
    assert output.shape == (1,)
    assert 0.5 <= output.item() <= 5.0


def test_model_different_users_get_different_predictions():
    """Different users should get different predicted ratings for same movie."""
    model = MatrixFactorization(num_users=10, num_movies=10, embedding_dim=16)
    movie_id = torch.LongTensor([0])
    pred_user0 = model(torch.LongTensor([0]), movie_id).item()
    pred_user1 = model(torch.LongTensor([1]), movie_id).item()
    # With random initialization, these should almost certainly be different
    assert pred_user0 != pred_user1


def test_model_embeddings_correct_size():
    """Embedding tables should have the right dimensions."""
    num_users, num_movies, embedding_dim = 100, 200, 32
    model = MatrixFactorization(num_users, num_movies, embedding_dim)
    assert model.user_embeddings.weight.shape == (num_users, embedding_dim)
    assert model.movie_embeddings.weight.shape == (num_movies, embedding_dim)
