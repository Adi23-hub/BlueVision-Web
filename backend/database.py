# In backend/database.py

import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# This will create a file named 'miniproject.db' in your main project folder
# Correctly computes the path to the 'MINI PROJECT' root
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'miniproject.db')}"

# Standard SQLAlchemy setup
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# --- DEFINE YOUR USER TABLE ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# --- A helper function to create the tables ---
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)