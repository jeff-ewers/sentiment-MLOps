
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import Session 
from datetime import datetime, timedelta
from typing import List, Optional
from app.database import Base

class DriftChangePoint(Base):
    """Model for storing detected drift changepoints"""
    __tablename__ = "drift_changepoints"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    confidence = Column(Float, nullable=False)
    metric = Column(String, nullable=False)
    before_params = Column(JSON, nullable=False)
    after_params = Column(JSON, nullable=False)
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# TODO: add drift repository class