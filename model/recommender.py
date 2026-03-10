import torch
import torch.nn as nn
 
class MatrixFactorization(nn.Module):
    """
    Netflix-style recommendation model.
    Learns embeddings (fingerprints) for users and movies,
    then predicts ratings by computing their dot product.
    """
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(MatrixFactorization, self).__init__()
 
        # Each user gets a fingerprint of 'embedding_dim' numbers
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
 
        # Each movie gets a fingerprint of 'embedding_dim' numbers
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
 
        # Bias: some users always rate high; some movies are just popular
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
 
        # Start with small random values
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
 
    def forward(self, user_ids, movie_ids):
        """Predict ratings for a batch of (user, movie) pairs."""
        user_embed  = self.user_embeddings(user_ids)
        movie_embed = self.movie_embeddings(movie_ids)
        user_b      = self.user_bias(user_ids).squeeze()
        movie_b     = self.movie_bias(movie_ids).squeeze()
 
        # Dot product: sum of element-wise multiplication
        dot = (user_embed * movie_embed).sum(dim=1)
 
        # Return raw score for training so gradients are not blocked by clipping.
        return dot + user_b + movie_b

    def predict(self, user_ids, movie_ids, min_rating=0.5, max_rating=5.0):
        """Inference helper: clip predictions to the valid rating range."""
        return torch.clamp(self.forward(user_ids, movie_ids), min_rating, max_rating)