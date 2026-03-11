import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.recommender import MatrixFactorization  # noqa: E402


class RatingsDataset(Dataset):
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


def train_model(
    csv_path='data/ratings.csv',
    model_save_path='model/trained_model.pt',
    embedding_dim=128,
    epochs=100,
    batch_size=1024,
    learning_rate=0.005,
):
    # 1. Load MovieLens data (uses 'userId' and 'movieId' column names)
    print('Loading MovieLens data...')
    df = pd.read_csv(csv_path)
    df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
    print(f'  {len(df):,} ratings | {df.user_id.nunique()} users | {df.movie_id.nunique()} movies')

    # 2. Remap IDs to sequential integers (0, 1, 2, ...)
    #    MovieLens movieIds go up to 193,609 — we cannot use them as array indexes
    user_enc = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    movie_enc = {m: i for i, m in enumerate(sorted(df.movie_id.unique()))}
    df['user_idx'] = df.user_id.map(user_enc)
    df['movie_idx'] = df.movie_id.map(movie_enc)

    # 3. Split 80% train / 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # 4. Create DataLoaders
    def make_loader(d, shuffle):
        ds = RatingsDataset(d.user_idx.values, d.movie_idx.values, d.rating.values)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    train_loader = make_loader(train_df, True)
    test_loader = make_loader(test_df, False)

    # 5. Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on: {device}')
    model = MatrixFactorization(len(user_enc), len(movie_enc), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # up from 1e-5
    loss_fn = nn.MSELoss()

    # 6. Training loop
    best_rmse = float('inf')
    print(f'Training for {epochs} epochs...')
    print('=' * 55)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for u, m, r in train_loader:
            u, m, r = u.to(device), m.to(device), r.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(u, m), r)
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for u, m, r in test_loader:
                u, m, r = u.to(device), m.to(device), r.to(device)
                test_loss += loss_fn(model(u, m), r).item()
        rmse = (test_loss / len(test_loader)) ** 0.5
        bar = '█' * int(20 * epoch / epochs) + '░' * (20 - int(20 * epoch / epochs))
        print(f'Epoch {epoch:2d}/{epochs} [{bar}] RMSE: {rmse:.4f}')
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'user_encoder': user_enc,
                'movie_encoder': movie_enc,
                'num_users': len(user_enc),
                'num_movies': len(movie_enc),
                'embedding_dim': embedding_dim,
            }, model_save_path)
    print('=' * 55)
    print(f'Best Test RMSE: {best_rmse:.4f}')
    print(f'Model saved to {model_save_path}')


if __name__ == '__main__':
    train_model()
