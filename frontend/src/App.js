import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function confidenceFromRating(rating) {
  const score = Number.isFinite(Number(rating)) ? Number(rating) : 0;
  return Math.max(0, Math.min(100, Math.round((score / 5) * 100)));
}

function whyRecommended(movie, topRatedMovies) {
  const genres = typeof movie?.genres === 'string' ? movie.genres.split('|') : [];
  if (!genres.length) return 'Similar users highly rated this title';

  const topGenres = new Set();
  topRatedMovies.forEach((item) => {
    if (typeof item?.genres === 'string') {
      item.genres.split('|').forEach((g) => topGenres.add(g));
    }
  });

  const overlap = genres.find((g) => topGenres.has(g));
  return overlap ? `Matches your interest in ${overlap}` : 'Matches your overall rating profile';
}

function PosterPlaceholder({ title }) {
  const letter = (title || '?').trim().charAt(0).toUpperCase() || '?';
  return (
    <div className='poster' aria-hidden='true'>
      <div className='poster-glow'></div>
      <span>{letter}</span>
    </div>
  );
}

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
  const confidence = confidenceFromRating(movie?.predicted_rating);

  return (
    <article className='movie-card'>
      <PosterPlaceholder title={movie?.title} />
      <div className='movie-main'>
        <div className='movie-topline'>
          <div className='movie-rank'>#{rank}</div>
          <span className='movie-confidence'>{confidence}% match</span>
        </div>
        <h3 className='movie-title'>{movie?.title || 'Untitled movie'}</h3>
        {genres.length > 0 ? (
          <div className='movie-genres'>
            {genres.map((g) => <span key={g} className='genre-tag'>{g}</span>)}
          </div>
        ) : (
          <div className='movie-genres'>
            <span className='genre-tag'>Unknown genre</span>
          </div>
        )}
        <div className='movie-rating-row'>
          <StarRating rating={movie?.predicted_rating} />
          <span className='why-badge' title={movie?.why_recommended || 'Recommended based on your profile'}>
            {movie?.why_recommended || 'Why recommended'}
          </span>
        </div>
      </div>
    </article>
  );
}

function MovieSkeletonCard() {
  return (
    <div className='movie-card skeleton'>
      <div className='poster shimmer'></div>
      <div className='movie-main'>
        <div className='line shimmer w40'></div>
        <div className='line shimmer w85'></div>
        <div className='line shimmer w70'></div>
        <div className='line shimmer w55'></div>
      </div>
    </div>
  );
}

export default function App() {
  const [userId, setUserId] = useState('');
  const [recommendations, setRecs] = useState([]);
  const [rawRecommendations, setRawRecommendations] = useState([]);
  const [topRatedMovies, setTopRatedMovies] = useState([]);
  const [trendingMovies, setTrendingMovies] = useState([]);
  const [selectedGenre, setSelectedGenre] = useState('All');
  const [loading, setLoading] = useState(false);
  const [trendingLoading, setTrendingLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasSearched, setHasSearched] = useState(false);

  useEffect(() => {
    const fetchTrending = async () => {
      setTrendingLoading(true);
      try {
        const res = await axios.get(`${API_URL}/popular?top_k=8`);
        const list = Array.isArray(res.data?.movies) ? res.data.movies : [];
        setTrendingMovies(list);
      } catch {
        setTrendingMovies([]);
      } finally {
        setTrendingLoading(false);
      }
    };

    fetchTrending();
  }, []);

  const availableGenres = useMemo(() => {
    const all = new Set(['All']);
    rawRecommendations.forEach((movie) => {
      if (typeof movie?.genres === 'string') {
        movie.genres.split('|').forEach((g) => all.add(g));
      }
    });
    return Array.from(all);
  }, [rawRecommendations]);

  const filteredRecommendations = useMemo(() => {
    if (selectedGenre === 'All') return recommendations;
    return recommendations.filter((movie) => typeof movie?.genres === 'string' && movie.genres.includes(selectedGenre));
  }, [recommendations, selectedGenre]);

  const getRecommendations = async () => {
    if (!userId) { setError('Please enter a User ID'); return; }
    if (userId < 1 || userId > 610) {
      setError('User ID must be between 1 and 610'); return;
    }
    setLoading(true);
    setError('');
    try {
      const [recsRes, topRatedRes] = await Promise.all([
        axios.get(`${API_URL}/recommend/${userId}?top_k=12`),
        axios.get(`${API_URL}/users/${userId}/top-rated?top_k=5`),
      ]);

      const res = recsRes;
      const contentType = String(res.headers?.['content-type'] || '').toLowerCase();
      if (contentType.includes('text/html')) {
        setRecs([]);
        setRawRecommendations([]);
        setTopRatedMovies([]);
        setHasSearched(true);
        setError('API URL is pointing to a web page, not the backend API. Set REACT_APP_API_URL to your deployed FastAPI URL.');
        return;
      }

      const nextRecs = Array.isArray(res.data?.recommendations) ? res.data.recommendations : [];
      const contextMovies = Array.isArray(topRatedRes.data?.movies) ? topRatedRes.data.movies : [];

      if (!Array.isArray(res.data?.recommendations)) {
        setError('Unexpected API response format (missing recommendations array).');
      }

      const enriched = nextRecs.map((movie) => ({
        ...movie,
        confidence: confidenceFromRating(movie?.predicted_rating),
        why_recommended: whyRecommended(movie, contextMovies),
      }));

      setRawRecommendations(enriched);
      setRecs(enriched);
      setTopRatedMovies(contextMovies);
      setSelectedGenre('All');
      setHasSearched(true);
    } catch (err) {
      setError('Could not fetch recommendations right now. Please retry in a moment.');
    } finally {
      setLoading(false);
    }
  };

  const pickRandomUser = () => {
    const randomId = Math.floor(Math.random() * 610) + 1;
    setUserId(String(randomId));
    setError('');
  };

  return (
    <div className='app cinematic-bg'>
      <div className='bg-layer'></div>
      <nav className='navbar'>
        <div className='brand'>
          <span className='brand-mark'>AI</span>
          <div>
            <h1>Movie Recommender</h1>
            <p>Personalized picks in seconds</p>
          </div>
        </div>
        <a href='https://github.com/ibragot/ai-recommender-system' target='_blank' rel='noreferrer'>GitHub</a>
      </nav>

      <main className='layout'>
        <section className='left-panel'>
          <div className='panel hero-panel'>
            <h2>Find movies for any user</h2>
            <p>Enter a user ID to generate ranked recommendations with confidence and genre context.</p>

            <label htmlFor='user-id'>User ID (1-610)</label>
            <div className='search-row'>
              <input
                id='user-id'
                type='number'
                min='1'
                max='610'
                placeholder='e.g. 42'
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && getRecommendations()}
              />
              <button onClick={getRecommendations} disabled={loading}>
                {loading ? 'Loading...' : 'Get Recommendations'}
              </button>
            </div>

            <div className='action-row'>
              <button className='secondary-btn' onClick={pickRandomUser}>Try a random user</button>
              <div className='genre-filter'>
                <label htmlFor='genre'>Genre</label>
                <select id='genre' value={selectedGenre} onChange={(e) => setSelectedGenre(e.target.value)}>
                  {availableGenres.map((g) => (
                    <option key={g} value={g}>{g}</option>
                  ))}
                </select>
              </div>
            </div>

            {error && (
              <div className='error-card'>
                <p>{error}</p>
                <button className='retry-btn' onClick={getRecommendations}>Retry</button>
              </div>
            )}

            <p className='hint'>Tip: Try users 1, 42, 100, 250, or 500.</p>
          </div>

          <div className='panel context-panel'>
            <h3>User Top-Rated Movies</h3>
            {loading ? (
              <div className='skeleton-list'>
                {[...Array(4)].map((_, i) => <div key={i} className='line shimmer w85'></div>)}
              </div>
            ) : topRatedMovies.length > 0 ? (
              <ul>
                {topRatedMovies.map((movie) => (
                  <li key={`top-${movie.id}`}>
                    <span>{movie.title}</span>
                    <strong>{Number(movie.rating || 0).toFixed(1)}</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p className='muted'>Search a user to see their highest-rated titles.</p>
            )}
          </div>
        </section>

        <aside className='right-panel'>
          <div className='panel trending-panel'>
            <h3>Trending</h3>
            {trendingLoading ? (
              <div className='skeleton-list'>
                {[...Array(6)].map((_, i) => <div key={i} className='line shimmer w70'></div>)}
              </div>
            ) : trendingMovies.length > 0 ? (
              <ul>
                {trendingMovies.map((movie, i) => (
                  <li key={`trend-${movie.id}`}>
                    <span>{i + 1}. {movie.title}</span>
                    <strong>{Number(movie.avg_rating || 0).toFixed(2)}</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p className='muted'>Trending data unavailable right now.</p>
            )}
          </div>
        </aside>
      </main>

      <section className='results-wrap'>
        <div className='results-head'>
          <h2>Recommendations</h2>
          {hasSearched && <span>{filteredRecommendations.length} movies shown</span>}
        </div>

        {loading && (
          <div className='movie-grid'>
            {[...Array(6)].map((_, i) => <MovieSkeletonCard key={i} />)}
          </div>
        )}

        {!loading && hasSearched && filteredRecommendations.length > 0 && (
          <div className='movie-grid'>
            {filteredRecommendations.map((movie, i) => (
                <MovieCard key={movie.id} movie={movie} rank={i + 1} />
            ))}
          </div>
        )}

        {!loading && hasSearched && filteredRecommendations.length === 0 && (
          <div className='empty-state'>
            <p>No movies match this filter yet. Try another genre.</p>
          </div>
        )}
      </section>

      <footer className='footer'>
        <p>Built with PyTorch, FastAPI, PostgreSQL, and Docker</p>
      </footer>
    </div>
  );
}