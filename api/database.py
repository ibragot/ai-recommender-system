import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, BigInteger, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://recommender:password@db:5432/recommender_db'
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)


class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    genres = Column(String)


class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    movie_id = Column(Integer, index=True)
    rating = Column(Float)
    timestamp = Column(BigInteger, nullable=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
    print('Database tables created')


def seed_database():
    """Load MovieLens CSV files into PostgreSQL."""
    db = SessionLocal()
    try:
        if db.execute(text('SELECT COUNT(*) FROM users')).scalar() > 0:
            print('Database already seeded, skipping')
            return
        print('Seeding database with MovieLens data...')

        # Load movies (MovieLens uses 'movieId' with capital I)
        movies = pd.read_csv('data/movies.csv').rename(columns={'movieId': 'movie_id'})
        for _, r in movies.iterrows():
            db.add(Movie(id=int(r.movie_id), title=str(r.title), genres=str(r.genres)))
        db.commit()
        print(f'  {len(movies):,} movies loaded')

        # Load ratings (MovieLens uses 'userId' and 'movieId')
        ratings = pd.read_csv('data/ratings.csv').rename(
            columns={'userId': 'user_id', 'movieId': 'movie_id'})

        # Create users from unique user IDs in ratings file
        for uid in sorted(ratings.user_id.unique()):
            db.add(User(id=int(uid), username=f'user_{uid}'))
        db.commit()
        print(f'  {ratings.user_id.nunique()} users loaded')

        # Load ratings in chunks of 5000 (100K rows takes time)
        chunk = 5000
        for i in range(0, len(ratings), chunk):
            for _, r in ratings.iloc[i:i + chunk].iterrows():
                db.add(Rating(user_id=int(r.user_id), movie_id=int(r.movie_id),
                              rating=float(r.rating), timestamp=int(r.timestamp)))
            db.commit()
            print(f'  Ratings: {min(i+chunk, len(ratings)):,}/{len(ratings):,}', end='\r')
        print(f'  {len(ratings):,} ratings loaded')
        print('Database seeding complete!')
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
