import os, sys, torch
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
from contextlib import asynccontextmanager
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.database import create_tables, seed_database, get_db, User, Movie, Rating
from model.recommender import MatrixFactorization
from evaluation.metrics import rmse, mae
 
MODEL_PATH   = os.getenv('MODEL_PATH', 'model/trained_model.pt')
model_state  = {'model': None, 'user_enc': None, 'movie_enc': None, 'movie_dec': None}
 
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f'No model at {MODEL_PATH} — run train.py first')
        return
    ck = torch.load(MODEL_PATH, map_location='cpu')
    m  = MatrixFactorization(ck['num_users'], ck['num_movies'], ck['embedding_dim'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    model_state.update({
        'model':     m,
        'user_enc':  ck['user_encoder'],
        'movie_enc': ck['movie_encoder'],
        'movie_dec': {v: k for k, v in ck['movie_encoder'].items()},
    })
    print('Model loaded successfully')
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables(); seed_database(); load_model()
    yield
 
app = FastAPI(
    title='Movie Recommendation API',
    description='Netflix-style recommendations via Matrix Factorization',
    version='1.0.0',
    lifespan=lifespan,
)
 
class RatingCreate(BaseModel):
    user_id: int; movie_id: int; rating: float
 
@app.get('/')
def root():
    return {'status': 'ok', 'model_loaded': model_state['model'] is not None}
 
@app.get('/users')
def get_users(db: Session = Depends(get_db)):
    return [{'id': u.id, 'username': u.username} for u in db.query(User).limit(50).all()]
 
@app.get('/movies')
def get_movies(db: Session = Depends(get_db)):
    return [{'id': m.id, 'title': m.title, 'genres': m.genres} for m in db.query(Movie).limit(50).all()]
 
@app.post('/ratings')
def add_rating(data: RatingCreate, db: Session = Depends(get_db)):
    if not 1.0 <= data.rating <= 5.0:
        raise HTTPException(400, 'Rating must be 1.0-5.0')
    db.add(Rating(user_id=data.user_id, movie_id=data.movie_id, rating=data.rating))
    db.commit()
    return {'message': 'Rating added', 'data': data}
 
@app.get('/recommend/{user_id}')
def recommend(user_id: int, top_k: int = 10, db: Session = Depends(get_db)):
    if model_state['model'] is None:
        raise HTTPException(503, 'Model not loaded. Run train.py first.')
    user = db.query(User).filter(User.id == user_id).first()
    if not user: raise HTTPException(404, f'User {user_id} not found')
 
    rated = {r.movie_id for r in db.query(Rating).filter(Rating.user_id==user_id).all()}
    user_enc, movie_enc = model_state['user_enc'], model_state['movie_enc']
 
    if user_id not in user_enc:
        popular = db.query(Movie).limit(top_k).all()
        return {'user_id': user_id, 'method': 'popularity_fallback',
                'recommendations': [{'id': m.id, 'title': m.title} for m in popular]}
 
    u_idx = user_enc[user_id]
    scores = []
    for movie in db.query(Movie).all():
        if movie.id in rated or movie.id not in movie_enc: continue
        m_idx = movie_enc[movie.id]
        with torch.no_grad():
            pred = model_state['model'].predict(torch.LongTensor([u_idx]),
                                                torch.LongTensor([m_idx])).item()
        scores.append({'id': movie.id, 'title': movie.title,
                       'genres': movie.genres, 'predicted_rating': round(pred, 2)})
 
    scores.sort(key=lambda x: x['predicted_rating'], reverse=True)
    return {'user_id': user_id, 'username': user.username,
            'method': 'matrix_factorization', 'recommendations': scores[:top_k]}
 
@app.get('/metrics')
def get_metrics(db: Session = Depends(get_db)):
    if model_state['model'] is None:
        raise HTTPException(503, 'Model not loaded')
    ratings = db.query(Rating).limit(2000).all()  # Sample for speed
    user_enc, movie_enc = model_state['user_enc'], model_state['movie_enc']
    actuals, preds = [], []
    for r in ratings:
        if r.user_id not in user_enc or r.movie_id not in movie_enc: continue
        with torch.no_grad():
            p = model_state['model'].predict(torch.LongTensor([user_enc[r.user_id]]),
                                             torch.LongTensor([movie_enc[r.movie_id]])).item()
        actuals.append(r.rating); preds.append(p)
    return {'rmse': round(rmse(actuals, preds), 4),
            'mae':  round(mae(actuals, preds), 4),
            'ratings_evaluated': len(actuals),
            'model': 'Matrix Factorization (PyTorch)'}