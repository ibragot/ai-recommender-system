# 🎬 AI Movie Recommendation Engine

![CI Pipeline](https://github.com/ibragot/ai-recommender-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![Live](https://img.shields.io/badge/Live-airecommender.noureldin.dev-brightgreen)

> **Netflix-style personalized movie recommendations powered by Matrix Factorization.**
> Trained on 100,836 real ratings from the MovieLens research dataset.
> Live at **[airecommender.noureldin.dev](https://airecommender.noureldin.dev)**

---

## 🎯 What It Does

This system learns each user's unique taste from their rating history and predicts which movies they will enjoy — even movies they have never seen. It solves a fundamental problem: the user-movie matrix is **98.3% empty** (most users have not rated most movies), and the AI fills in those gaps intelligently.

A user visiting the site picks their ID, clicks a button, and instantly receives a ranked list of personalized movie recommendations — each with a predicted rating and a match percentage showing how confident the model is.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────────────┐
                        │         Developer (You)          │
                        │   writes code, trains model      │
                        └──────────────┬──────────────────┘
                                       │ git push
                         ┌─────────────▼─────────────┐
                         │        GitHub              │
                         │   stores source code       │
                         └──────┬──────────┬──────────┘
                                │ CI       │ CD
                    ┌───────────▼──┐   ┌───▼────────────────┐
                    │ GitHub       │   │ Railway            │
                    │ Actions      │   │ (Backend Server)   │
                    │              │   │                    │
                    │ • Run tests  │   │ • FastAPI API      │
                    │ • Build      │   │ • PostgreSQL DB    │
                    │   Docker     │   │ • Downloads model  │
                    │ • Check code │   │   from HuggingFace │
                    └──────────────┘   └─────────┬──────────┘
                                                 │ serves
                              ┌──────────────────▼──────────────────┐
                              │         Hugging Face Hub             │
                              │    stores trained_model.pt           │
                              │    (downloaded on server startup)    │
                              └──────────────────┬──────────────────┘
                                                 │
                    ┌────────────────────────────▼────────────────────────┐
                    │                    Vercel                           │
                    │              React Frontend                         │
                    │        airecommender.noureldin.dev                  │
                    └────────────────────────────┬────────────────────────┘
                                                 │ user visits
                              ┌──────────────────▼──────────────────┐
                              │              End User                │
                              │   picks a user ID → sees            │
                              │   personalized movie recommendations  │
                              └─────────────────────────────────────┘
```

---

## 🦋 Data Flow — How a Recommendation Is Generated

```
                              User clicks
                         "Get Recommendations"
                                  │
                    ┌─────────────▼─────────────┐
                    │      React Frontend        │
                    │   GET /recommend/{user_id} │
                    └─────────────┬─────────────┘
                                  │ HTTP request
                    ┌─────────────▼─────────────┐
                    │       FastAPI Backend      │
                    │                           │
                    │  1. Look up user in DB    │
                    │  2. Get rated movie IDs   │
                    └──────┬──────────┬─────────┘
                           │          │
              ┌────────────▼──┐   ┌───▼────────────┐
              │  PostgreSQL   │   │  PyTorch Model  │
              │               │   │                 │
              │ • Users       │   │ • User embedding│
              │ • Movies      │   │ • Movie embedding│
              │ • Ratings     │   │ • Dot product   │
              └────────────┬──┘   └───┬─────────────┘
                           │          │
                    ┌──────▼──────────▼──────┐
                    │   Score every unrated  │
                    │   movie for this user  │
                    │   Sort by predicted    │
                    │   rating descending    │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │   Top 10 returned      │
                    │   as JSON response     │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │   Frontend renders     │
                    │   movie cards with     │
                    │   stars + match %      │
                    └────────────────────────┘
```

---

## 🧠 The AI Model — Matrix Factorization

The recommendation engine uses **Matrix Factorization**, the same technique used by winners of the Netflix Prize competition.

### The Core Problem

Every user has rated only a tiny fraction of available movies. Out of 5,931,640 possible user-movie combinations in the dataset, only 100,836 have ratings — meaning **98.3% of the matrix is empty**.

```
           Toy Story   Pulp Fiction   The Matrix   Titanic   ...9,738 more
User 1        4.0          ???            5.0         ???
User 2        ???          5.0            ???         3.0
User 3        2.0          ???            ???         ???
User 610      ???          ???            4.5         ???
```

### How Matrix Factorization Solves It

Each user gets a learned **taste fingerprint** — a vector of 64 numbers that captures their preferences. Each movie gets a learned **style fingerprint** — a vector of 64 numbers capturing its characteristics. To predict how much a user will like a movie, the model computes the dot product of their fingerprints.

```
User 1 fingerprint:  [0.82, -0.31, 0.94, 0.17, ...]  (64 numbers)
Matrix fingerprint:  [0.76,  0.88, 0.91, 0.22, ...]  (64 numbers)
                              ↓ dot product
              Predicted rating: 4.79 ⭐
```

The model learns these fingerprints by studying 100,836 real ratings over 20 training epochs, adjusting the numbers each time it makes a wrong prediction until the error is minimized.

### Training Results

| Metric | Value |
|--------|-------|
| Test RMSE | ~0.87 |
| Test MAE | ~0.67 |
| Training epochs | 100 |
| Embedding dimension | 128 |
| Optimizer | Adam (weight decay 1e-5) |
| Batch size | 1,024 |

An RMSE of ~0.87 means predictions are off by less than 1 star on average. Netflix's prize-winning algorithm achieved ~0.85 on a significantly harder version of this problem.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| AI Model | PyTorch | Matrix Factorization neural network |
| Data Processing | NumPy, Pandas | Loading and encoding the dataset |
| Backend API | FastAPI, Uvicorn | REST endpoints for recommendations |
| Database | PostgreSQL, SQLAlchemy | Storing users, movies, ratings |
| Model Storage | Hugging Face Hub | Hosting the trained model file |
| Frontend | React | User interface |
| Backend Hosting | Railway | Running FastAPI + PostgreSQL |
| Frontend Hosting | Vercel | Serving the React app |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Containerization | Docker, Docker Compose | Consistent environments |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check — model loaded status |
| GET | `/users` | List users in the database |
| GET | `/movies` | List movies in the database |
| POST | `/ratings` | Submit a new user rating |
| GET | `/recommend/{user_id}` | Get top-K personalized recommendations |
| GET | `/recommend/{user_id}?top_k=5` | Limit to top 5 recommendations |
| GET | `/metrics` | Live RMSE and MAE model evaluation scores |

### Example Response

```bash
GET /recommend/42?top_k=3
```

```json
{
  "user_id": 42,
  "username": "user_42",
  "method": "matrix_factorization",
  "recommendations": [
    {
      "id": 318,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "predicted_rating": 4.87
    },
    {
      "id": 296,
      "title": "Pulp Fiction (1994)",
      "genres": "Comedy|Crime|Drama|Thriller",
      "predicted_rating": 4.82
    },
    {
      "id": 2571,
      "title": "Matrix, The (1999)",
      "genres": "Action|Sci-Fi|Thriller",
      "predicted_rating": 4.79
    }
  ]
}
```

---

## 📊 Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **RMSE** | Average prediction error in stars. Lower is better. 0 = perfect. |
| **MAE** | Mean absolute error. Less sensitive to outliers than RMSE. |
| **Precision@K** | Of top K recommendations, what % did the user actually like? |
| **Recall@K** | Of all movies the user likes, what % did we surface in top K? |
| **NDCG@K** | Did the best recommendations appear at the very top? 1.0 = perfect. |

---

## 🗂️ Project Structure

```
ai-recommender-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml              ← CI/CD pipeline (test → build → deploy)
│
├── api/
│   ├── main.py                 ← FastAPI routes, model loading, HuggingFace download
│   └── database.py             ← PostgreSQL tables and data seeding
│
├── model/
│   ├── recommender.py          ← PyTorch Matrix Factorization model definition
│   └── train.py                ← Training script
│
├── evaluation/
│   └── metrics.py              ← RMSE, MAE, Precision@K, Recall@K, NDCG@K
│
├── frontend/
│   └── src/
│       ├── App.js              ← Main React component
│       └── App.css             ← Styling
│
├── tests/
│   ├── test_metrics.py         ← Unit tests for evaluation metrics
│   └── test_model.py           ← Unit tests for PyTorch model
│
├── docker/
│   ├── dockerfile              ← Container build instructions
│   └── docker-compose.yml      ← Local development environment
│
├── data/                       ← MovieLens CSV files (not committed)
├── requirements.txt
└── README.md
```

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers this automated pipeline:

```
git push
    │
    ├── Run Tests          (pytest — metrics and model unit tests)
    ├── Code Quality       (flake8 linter)
    ├── Build Docker       (verifies container compiles correctly)
    ├── Deploy Backend     (Railway — FastAPI + PostgreSQL)
    └── Deploy Frontend    (Vercel — React app)
```

The backend downloads the trained model from Hugging Face Hub automatically on startup — no model file is ever committed to the repository.

---

## 🗺️ Roadmap

- [ ] Neural Collaborative Filtering (NCF) model for improved accuracy
- [ ] Content-based filtering using movie genres and metadata
- [ ] User registration and personal rating submission
- [ ] A/B testing framework to compare model versions live
- [ ] Retrain pipeline triggered automatically by new ratings
- [ ] Upgrade to full MovieLens 25M dataset (25 million ratings)

---

## 📚 References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/) — GroupLens Research, University of Minnesota
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) — Koren, Bell, Volinsky (Netflix Prize paper)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 👤 Author

**Noureldin Ibrahim**
- 🌐 [noureldin.dev](https://noureldin.dev)
- 🎬 [Live Demo](https://airecommender.noureldin.dev)
- 💻 [GitHub](https://github.com/ibragot)

---

*A full-stack production ML system — from raw data to live deployment.*
