import os
import sys
import torch
import html
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.database import create_tables, seed_database, get_db, SessionLocal, User, Movie, Rating  # noqa: E402
from model.recommender import MatrixFactorization  # noqa: E402
from evaluation.metrics import rmse, mae  # noqa: E402
MODEL_PATH = os.getenv('MODEL_PATH', 'model/trained_model.pt')
model_state = {'model': None, 'user_enc': None, 'movie_enc': None, 'movie_dec': None}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f'No model at {MODEL_PATH} — run train.py first')
        return
    ck = torch.load(MODEL_PATH, map_location='cpu')
    m = MatrixFactorization(ck['num_users'], ck['num_movies'], ck['embedding_dim'])
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    model_state.update({
        'model': m,
        'user_enc': ck['user_encoder'],
        'movie_enc': ck['movie_encoder'],
        'movie_dec': {v: k for k, v in ck['movie_encoder'].items()},
    })
    print('Model loaded successfully')


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    seed_database()
    load_model()
    yield


app = FastAPI(
    title='Movie Recommendation API',
    description='Netflix-style recommendations via Matrix Factorization',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the Images folder as static files
images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Images')
if os.path.isdir(images_dir):
    app.mount('/static/images', StaticFiles(directory=images_dir), name='images')

@app.get('/favicon.ico', include_in_schema=False)
def favicon():
    ico_path = os.path.join(images_dir, 'favicon.ico')
    if os.path.exists(ico_path):
        return FileResponse(ico_path, media_type='image/x-icon')
    jpg_path = os.path.join(images_dir, 'default-avatar-icon-of-social-media-user-vector.jpg')
    return FileResponse(jpg_path, media_type='image/jpeg')


class RatingCreate(BaseModel):
    user_id: int
    movie_id: int
    rating: float


@app.get('/')
def root():
    # ── Server-side render users so the page NEVER appears blank ──
    users_html = ''
    try:
        db = SessionLocal()
        users = db.query(User).limit(50).all()
        db.close()
        for u in users:
            safe_name = html.escape(u.username or 'User')
            users_html += (
                f'<div class="user-card" onclick="selectUser({u.id}, this)">'
                f'<svg class="avatar" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
                f'<rect width="64" height="64" rx="32" fill="#e0e7ff"/>'
                f'<circle cx="32" cy="24" r="10" fill="#6366f1"/>'
                f'<ellipse cx="32" cy="52" rx="18" ry="14" fill="#6366f1"/>'
                f'</svg>'
                f'<div class="name">{safe_name}</div>'
                f'<div class="uid">ID: {u.id}</div></div>'
            )
    except Exception:
        users_html = '<div class="placeholder">Could not load users — is the database running?</div>'

    if not users_html:
        users_html = '<div class="placeholder">No users found. Seed the database first.</div>'

    return HTMLResponse(content=f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movie Recommender</title>
<link rel="icon" href="/favicon.ico" sizes="any">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,'Segoe UI',system-ui,Roboto,Helvetica,Arial,sans-serif;background:#f0f2f5;color:#1e293b;min-height:100vh}}
header{{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#312e81 100%);padding:0;position:relative;overflow:hidden}}
header::before{{content:"";position:absolute;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse at 20% 50%,rgba(99,102,241,.15) 0%,transparent 50%),radial-gradient(ellipse at 80% 20%,rgba(139,92,246,.1) 0%,transparent 50%);pointer-events:none}}
.header-inner{{max-width:1020px;margin:0 auto;padding:2rem 2rem 1.8rem;position:relative;z-index:1;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem}}
.header-brand{{display:flex;align-items:center;gap:1rem}}
.header-logo{{width:48px;height:48px;background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;box-shadow:0 4px 16px rgba(99,102,241,.35)}}
.header-text h1{{font-size:1.5rem;font-weight:700;color:#fff;letter-spacing:-.02em}}
.header-text h1 .accent{{background:linear-gradient(90deg,#a5b4fc,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.header-text p{{color:#94a3b8;font-size:.82rem;margin-top:.25rem;font-weight:400}}
.header-text p .dot{{display:inline-block;width:5px;height:5px;background:#22c55e;border-radius:50%;margin-right:.35rem;vertical-align:middle;box-shadow:0 0 6px rgba(34,197,94,.5)}}
.header-nav{{display:flex;gap:.5rem}}
.header-nav a{{color:#c7d2fe;text-decoration:none;font-size:.78rem;font-weight:500;padding:.4rem .8rem;border:1px solid rgba(199,210,254,.2);border-radius:8px;transition:all .2s;backdrop-filter:blur(4px);background:rgba(255,255,255,.04)}}
.header-nav a:hover{{background:rgba(99,102,241,.3);border-color:rgba(199,210,254,.4);color:#fff}}
.header-bar{{height:3px;background:linear-gradient(90deg,#6366f1,#8b5cf6,#a78bfa,#6366f1);background-size:200% 100%;animation:shimmer 3s linear infinite}}
@keyframes shimmer{{0%{{background-position:200% 0}}100%{{background-position:-200% 0}}}}
.container{{max-width:1020px;margin:0 auto;padding:1.5rem 1.25rem}}
.step{{background:#fff;border-radius:10px;padding:1.4rem;margin-bottom:1.25rem;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.step-header{{display:flex;align-items:center;gap:.65rem;margin-bottom:1rem}}
.step-num{{background:#6366f1;color:#fff;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.8rem;flex-shrink:0}}
.step-header h2{{font-size:1rem;font-weight:600;color:#1e293b}}
.user-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:.65rem}}
.user-card{{background:#f8fafc;border:2px solid #e2e8f0;border-radius:10px;padding:.75rem .4rem;text-align:center;cursor:pointer;transition:.2s}}
.user-card:hover{{border-color:#6366f1;transform:translateY(-2px);box-shadow:0 4px 12px rgba(99,102,241,.12)}}
.user-card.selected{{border-color:#6366f1;background:#eef2ff}}
.user-card .avatar{{width:44px;height:44px;border-radius:50%;margin:0 auto .4rem;display:block}}
.user-card .name{{font-size:.78rem;color:#374151;font-weight:600}}
.user-card .uid{{font-size:.68rem;color:#9ca3af;margin-top:.1rem}}
#recommendations{{min-height:50px}}
.loading{{text-align:center;color:#94a3b8;padding:1.5rem}}
.spinner{{display:inline-block;width:20px;height:20px;border:3px solid #e2e8f0;border-top-color:#6366f1;border-radius:50%;animation:spin .7s linear infinite;margin-right:.4rem;vertical-align:middle}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.movie-list{{display:grid;gap:.55rem}}
.movie-card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:.9rem 1rem;display:flex;align-items:center;gap:.9rem;transition:.2s}}
.movie-card:hover{{border-color:#6366f1;box-shadow:0 2px 8px rgba(99,102,241,.08)}}
.movie-rank{{font-size:1.1rem;font-weight:700;color:#6366f1;min-width:30px;text-align:center}}
.movie-info{{flex:1}}
.movie-title{{font-size:.9rem;color:#1e293b;font-weight:600}}
.movie-genres{{font-size:.72rem;color:#94a3b8;margin-top:.15rem}}
.movie-rating{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:.35rem .65rem;text-align:center;min-width:90px}}
.movie-rating .stars{{display:flex;gap:1px;justify-content:center;margin-bottom:.15rem}}
.movie-rating .stars svg{{width:16px;height:16px}}
.movie-rating .score{{font-size:.88rem;font-weight:700;color:#6366f1;margin-top:.1rem}}
.movie-rating .label{{font-size:.58rem;color:#94a3b8;text-transform:uppercase}}
.placeholder{{text-align:center;color:#94a3b8;padding:2rem .8rem;font-size:.85rem}}
.error{{text-align:center;color:#ef4444;padding:1.2rem}}
</style>
</head>
<body>
<header>
<div class="header-inner">
<div class="header-brand">
<div class="header-logo">&#127916;</div>
<div class="header-text">
<h1>Movie <span class="accent">Recommender</span></h1>
<p><span class="dot"></span>AI powered personalized movie recommendations</p>
</div>
</div>
<nav class="header-nav">
<a href="/docs">API Docs</a>
<a href="/redoc">ReDoc</a>
<a href="/health">Health</a>
</nav>
</div>
<div class="header-bar"></div>
</header>
<div class="container">
<div class="step">
<div class="step-header"><div class="step-num">1</div><h2>Select a User</h2></div>
<div class="user-grid">{users_html}</div>
</div>
<div class="step">
<div class="step-header"><div class="step-num">2</div><h2>Recommended Movies</h2></div>
<div id="recommendations"><div class="placeholder">&#128073; Click a user above to see their top movie picks</div></div>
</div>
</div>
<script>
function starSvg(fill){{var gold="#f59e0b",grey="#e2e8f0";if(fill>=1)return '<svg viewBox="0 0 20 20"><polygon points="10,1 12.9,7 19.5,7.6 14.5,12 16,18.5 10,15 4,18.5 5.5,12 0.5,7.6 7.1,7" fill="'+gold+'"/></svg>';if(fill<=0)return '<svg viewBox="0 0 20 20"><polygon points="10,1 12.9,7 19.5,7.6 14.5,12 16,18.5 10,15 4,18.5 5.5,12 0.5,7.6 7.1,7" fill="'+grey+'"/></svg>';var id="s"+Math.random().toString(36).substr(2,5);return '<svg viewBox="0 0 20 20"><defs><linearGradient id="'+id+'"><stop offset="'+(fill*100)+'%" stop-color="'+gold+'"/><stop offset="'+(fill*100)+'%" stop-color="'+grey+'"/></linearGradient></defs><polygon points="10,1 12.9,7 19.5,7.6 14.5,12 16,18.5 10,15 4,18.5 5.5,12 0.5,7.6 7.1,7" fill="url(#'+id+')"/></svg>'}}
function starsHtml(r){{var h="";for(var i=0;i<5;i++){{var rem=r-i;if(rem>=1)h+=starSvg(1);else if(rem>0)h+=starSvg(rem);else h+=starSvg(0)}}return h}}
function selectUser(id,el){{document.querySelectorAll(".user-card").forEach(function(c){{c.classList.remove("selected")}});el.classList.add("selected");var r=document.getElementById("recommendations");r.innerHTML="<div class=\\"loading\\"><span class=\\"spinner\\"></span> Loading recommendations...</div>";var x=new XMLHttpRequest();x.open("GET","/recommend/"+id);x.onload=function(){{if(x.status===200){{var d=JSON.parse(x.responseText);if(!d.recommendations||!d.recommendations.length){{r.innerHTML="<div class=\\"placeholder\\">No recommendations for this user</div>";return}}var h="<div class=\\"movie-list\\">";for(var i=0;i<d.recommendations.length;i++){{var m=d.recommendations[i];h+="<div class=\\"movie-card\\"><div class=\\"movie-rank\\">#"+(i+1)+"</div><div class=\\"movie-info\\"><div class=\\"movie-title\\">"+m.title+"</div><div class=\\"movie-genres\\">"+(m.genres||"")+"</div></div><div class=\\"movie-rating\\"><div class=\\"stars\\">"+starsHtml(m.predicted_rating)+"</div><div class=\\"score\\">"+m.predicted_rating.toFixed(1)+" / 5.0</div><div class=\\"label\\">predicted</div></div></div>"}}h+="</div>";r.innerHTML=h}}else{{r.innerHTML="<div class=\\"error\\">Failed to load recommendations</div>"}}}}; x.onerror=function(){{r.innerHTML="<div class=\\"error\\">Network error</div>"}};x.send()}}
</script>
</body>
</html>''')

@app.get('/health')
def health():
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
    if not user:
        raise HTTPException(404, f'User {user_id} not found')

    rated = {r.movie_id for r in db.query(Rating).filter(Rating.user_id == user_id).all()}
    user_enc, movie_enc = model_state['user_enc'], model_state['movie_enc']

    if user_id not in user_enc:
        popular = db.query(Movie).limit(top_k).all()
        return {'user_id': user_id, 'method': 'popularity_fallback',
                'recommendations': [{'id': m.id, 'title': m.title} for m in popular]}

    u_idx = user_enc[user_id]
    scores = []
    for movie in db.query(Movie).all():
        if movie.id in rated or movie.id not in movie_enc:
            continue
        m_idx = movie_enc[movie.id]
        with torch.no_grad():
            pred = model_state['model'].predict(
                torch.LongTensor([u_idx]),
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
        if r.user_id not in user_enc or r.movie_id not in movie_enc:
            continue
        with torch.no_grad():
            p = model_state['model'].predict(
                torch.LongTensor([user_enc[r.user_id]]),
                torch.LongTensor([movie_enc[r.movie_id]])).item()
        actuals.append(r.rating)
        preds.append(p)
    return {'rmse': round(rmse(actuals, preds), 4),
            'mae': round(mae(actuals, preds), 4),
            'ratings_evaluated': len(actuals),
            'model': 'Matrix Factorization (PyTorch)'}
