from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictionRequest(BaseModel):
    """Validation schema for prediction requests"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description='Text to analyze for sentiment'
    )

    @validator('text')
    def text_must_be_valid(cls, v: str) -> str:
        """Additional validation for text field"""
        #strip whitespace
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty or whitespace")
        return v
    
class PredictionResponse(BaseModel):
    """Schema for prediction responses"""
    text: str
    sentiment: str = Field(..., description="Predicted sentiment (+/-)")
    confidence: float = Field(..., ge=0, le=1, description= "Model confidence score")
    model_version: str = Field(default='v1', description='Model version')

    model_config = {
            "json_schema_extra": {
                "example": {
                    "text": "This project is going well!",
                    "sentiment": "POSITIVE",
                    "confidence": 0.95,
                    "model_version": "v1"
                }
            }
        }