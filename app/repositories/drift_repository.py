
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import Session 
from datetime import datetime, timedelta
from typing import List, Optional
from app.database_base import Base

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


class DriftRepository:
    """Repository for handling drift detection results"""

    def __init__(self, db: Session):
        self.db = db

    def save_changepoint(self,
                         timestamp: datetime,
                         confidence: float,
                         metric: str,
                         before_params: dict,
                         after_params: dict,
                         model_version: Optional[str] = None) -> DriftChangePoint:
            """Save a detected changepoint"""
            changepoint = DriftChangePoint(
                 timestamp=timestamp,
                 confidence=confidence,
                 metric=metric,
                 before_params=before_params,
                 after_params=after_params,
                 model_version=model_version
            )
            self.db.add(changepoint)
            self.db.commit()
            self.db.refresh(changepoint)
            return changepoint

    def get_recent_changepoints(self,
                                hours: int = 168, #prior week
                                metric: Optional[str] = None,
                                model_version: str = None) -> List[DriftChangePoint]:
        """Get recent changepoints with optional filters"""
        query = self.db.query(DriftChangePoint)

        if metric:
            query = query.filter(DriftChangePoint.metric == metric)
        
        if model_version:
            query = query.filter(DriftChangePoint.model_version == model_version)

        start_time = datetime.utcnow() - timedelta(hours=hours)
        query = query.filter(DriftChangePoint.timestamp >= start_time)

        return query.order_by(DriftChangePoint.timestamp.desc()).all()