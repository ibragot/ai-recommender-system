import { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function StarRating({ rating }) {
  const numericRating = Number(rating);
  const safeRating = Number.isFinite(numericRating) ? numericRating : 0;
  const quarterStep = 0.25;
  const normalizedRating = Math.max(0, Math.min(5, Math.round(safeRating / quarterStep) * quarterStep));

  const getFillPercent = (starIndex) => {
    const starStart = starIndex;
    const starEnd = starIndex + 1;
    if (normalizedRating <= starStart) return 0;
    if (normalizedRating >= starEnd) return 100;
    return (normalizedRating - starStart) * 100;
  };

  return (
    <span className='stars'>
      {[0, 1, 2, 3, 4].map((starIndex) => (
        <span
          key={starIndex}
          className='star'
          style={{ '--fill': `${getFillPercent(starIndex)}%` }}
          aria-hidden='true'
        >
          ★
        </span>
      ))}
      <span className='rating-number'> {safeRating.toFixed(2)}</span>
    </span>
  );
}

function MovieCard({ movie, rank }) {
  const genres = typeof movie?.genres === 'string' ? movie.genres.split('|').slice(0, 3) : [];
  return (
    <div className='movie-card'>
      <div className='movie-rank'>#{rank}</div>
      <div className='movie-info'>
        <h3 className='movie-title'>{movie?.title || 'Untitled movie'}</h3>
        {genres.length > 0 && (
          <div className='movie-genres'>
            {genres.map(g => <span key={g} className='genre-tag'>{g}</span>)}
          </div>
        )}
        <StarRating rating={movie.predicted_rating} />
      </div>
    </div>
  );
}

export default function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  const getRecommendations = async () => {
    if (!userId) { setError('Please enter a User ID'); return; }
    if (userId < 1 || userId > 610) {
      setError('User ID must be between 1 and 610'); return;
    }
    setLoading(true);
    setError('');
    try {
      const res = await axios.get(`${API_URL}/recommend/${userId}?top_k=10`);
      const nextRecs = Array.isArray(res.data?.recommendations) ? res.data.recommendations : [];
      if (!Array.isArray(res.data?.recommendations)) {
        setError('Unexpected API response format.');
      }
      setRecs(nextRecs);
      setHasSearched(true);
    } catch (err) {
      setError('Could not fetch recommendations. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className='app'>
      <header className='header'>
        <h1>🎬 AI Movie Recommender</h1>
        <p>Netflix-style recommendations powered by Matrix Factorization</p>
      </header>

      <main className='main'>
        <div className='search-box'>
          <label>Enter a User ID (1 – 610)</label>
          <div className='search-row'>
            <input
              type='number'
              min='1'
              max='610'
              placeholder='e.g. 42'
              value={userId}
              onChange={e => setUserId(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && getRecommendations()}
            />
            <button onClick={getRecommendations} disabled={loading}>
              {loading ? 'Loading...' : 'Get Recommendations'}
            </button>
          </div>
          {error && <p className='error'>{error}</p>}
          <p className='hint'>
            Trained on 100,836 real ratings from the MovieLens dataset. Try users 1, 42, 100, 250, or 500.
          </p>
        </div>

        {loading && (
          <div className='loading'>
            <div className='spinner'></div>
            <p>AI is thinking...</p>
          </div>
        )}

        {!loading && hasSearched && Array.isArray(recommendations) && recommendations.length > 0 && (
          <div className='results'>
            <h2>Top 10 Movies for User {userId}</h2>
            <div className='movie-list'>
              {recommendations.map((movie, i) => (
                <MovieCard key={movie.id} movie={movie} rank={i + 1} />
              ))}
            </div>
          </div>
        )}

        {!loading && hasSearched && (!Array.isArray(recommendations) || recommendations.length === 0) && (
          <p className='no-results'>No recommendations found for this user.</p>
        )}
      </main>

      <footer className='footer'>
        <p>Built with PyTorch · FastAPI · PostgreSQL · Docker</p>
        <a href='https://github.com/ibragot/ai-recommender-system'
          target='_blank' rel='noreferrer'>View on GitHub</a>
      </footer>
    </div>
  );
}