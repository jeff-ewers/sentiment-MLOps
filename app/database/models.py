from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.database.base import Base
from app.repositories.drift_repository import DriftChangePoint
import os

#create db url
SQLALCHEMY_DATABASE_URL = "sqlite:///./mlops.db"

#create sqlalchemy engine
engine = create_engine(
  SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} #for sqlite
 )

#create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



class Prediction(Base):
    """Model for storing sentiment predictions"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_model_output = Column(JSON)
    request_metadata = Column(JSON)

#fx to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#create tables
def init_db():
    Base.metadata.create_all(bind=engine)