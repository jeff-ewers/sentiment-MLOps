from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Optional 

class ChangePointResponse(BaseModel):
    """Schema for drift detection changepoint"""
    timestamp: datetime
    confidence: float = Field(..., ge=0, le=1)
    metric: str 
    before_params: Dict
    after_params: Dict


class DriftDetectionRequest(BaseModel):
    """Schema for drift detection request"""
    window_size_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Analysis window size in hours"
    )
    min_probability: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Minimum probability threshold for drift detection"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to analyze"
    )