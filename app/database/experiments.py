from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.database.base import Base

class ExperimentStatus(enum.Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Experiment(Base):
    """Model for A/B testing experiments"""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    status = Column(Enum(ExperimentStatus), nullable=False, default=ExperimentStatus.DRAFT)
    created_at = Column(DateTime, default=datetime.utcnow())
    updated_at = Column(DateTime, default=datetime.utcnow(), onupdate=datetime.utcnow())
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    

    #statistical configuration
    minimum_sample_size = Column(Integer, nullable=False, default=1000)
    confidence_level = Column(Float, nullable=False, default=0.95)

    #relationships
    variants = relationship("ExperimentVariant", back_populates="experiment", cascade="all, delete-orphan")
    

class ExperimentVariant(Base):
    """Model for experiment variants/treatment"""
    __tablename__ = "experiment_variants"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    name = Column(String, nullable=False)
    config = Column(JSON, nullable=False)
    traffic_percentage = Column(Float, nullable=False)
    is_control = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow())

    #relationships
    experiment = relationship("Experiment", back_populates="variants")
    predictions = relationship("Prediction", back_populates="variant")

