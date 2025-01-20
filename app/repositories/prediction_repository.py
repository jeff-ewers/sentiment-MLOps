from sqlalchemy.orm import Session 
from datetime import datetime, timedelta
from typing import List, Optional
from ..database import Prediction

class PredictionRepository:
    """Repository for handling prediction DB operations"""

    def __init__(self, db: Session):
        self.db = db

    def create_prediction(self,
                          text: str,
                          sentiment: str,
                          confidence: float,
                          model_version: str,
                          raw_model_output: dict = None,
                          metadata: dict = None) -> Prediction:
        """Create a new prediction record"""
        db_prediction = Prediction(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            model_version=model_version,
            raw_model_output=raw_model_output,
            metadata=metadata
        )
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        return db_prediction
    
def get_recent_predictions(self, limit: int = 100) -> List[Prediction]:
    """Get most recent predictions"""
    return self.db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()

def get_predictions_by_timeframe(self, start_time: datetime, end_time: datetime) -> List[Prediction]:
    """Get predictions by specific timeframe"""
    return self.db.query(Prediction).filter(Prediction.created_at >= start_time).filter(Prediction.created_at <= end_time).order_by(Prediction.created_at.desc()).all()

def get_model_performance_stats(self, 
                                model_version: Optional[str] = None,
                                timeframe_hours: int = 24) -> dict:
    """Get performance statistics by model version"""
    query = self.db.query(Prediction)

    if model_version:
        query = query.filter(Prediction.model_version == model_version)

    start_time = datetime.utcnow() - timedelta(hours=timeframe_hours)
    query = query.filter(Prediction.created_at >= start_time)

    predictions = query.all()

    if not predictions:
        return {
            "total_predictions": 0,
            "average_confidence": 0,
            "sentiment_distribution": {"POSITIVE": 0, "NEGATIVE": 0}
        }
    
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0}
    total_confidence = 0

    for p in predictions:
        sentiment_counts[p.sentiment] += 1
        total_confidence += p.confidence

    return {
        "total_predictions": len(predictions),
        "average_confidence": total_confidence / len(predictions),
        "sentiment_distribution": sentiment_counts
    }