from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any
from datetime import datetime, timedelta

#request metrics
PREDICTION_REQUESTS = Counter(
    'sentiment_prediction_requests_total',
    'Total number of sentiment prediction requests',
    ['model_version'] 
)

PREDICTION_LATENCY = Histogram(
    'sentiment_prediction_latency_seconds',
    'Time spent processing prediction requests',
    ['model_version'] 
)

#model metrics
CONFIDENCE_SCORES = Histogram(
    'sentiment_confidence_scores',
    'Distribution of model confidence scores',
    ['model_version', 'predicted_sentiment'] 
)

SENTIMENT_PREDICTIONS = Counter(
    'sentiment_predictions_total',
    'Total predictions by sentiment class',
    ['model_version', 'sentiment'] 
)

#drift metrics
DRIFT_DETECTION_RUNS = Counter(
    'drift_detection_runs_total',
    'Number of drift detection runs',
    ['model_version'] 
)

DETECTED_DRIFTS = Counter(
    'detected_drifts_total',
    'Number of detected drift events',
    ['model_version', 'drift_type'] 
)

class MetricsTracker:
    """Centralized metrics tracking for the sentiment analysis service"""

    @staticmethod
    def track_prediction(model_version: str,
                         sentiment: str,
                         confidence: float,
                         latency: float):
        """Track a single prediction event"""
        PREDICTION_REQUESTS.labels(model_version=model_version).inc()
        PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)
        CONFIDENCE_SCORES.labels(model_version=model_version, predicted_sentiment=sentiment).observe(confidence)
        SENTIMENT_PREDICTIONS.labels(model_version=model_version, sentiment=sentiment).inc()

    @staticmethod
    def track_drift_detection(model_version: str,
                              detected_drifts: int = 0,
                              drift_types: Dict[str, Any] = None):
        """Track drift detections run and any detected drifts"""
        DRIFT_DETECTION_RUNS.labels(model_version=model_version).inc()

        if detected_drifts > 0 and drift_types:
            for drift_type, details in drift_types.items():
                DETECTED_DRIFTS.labels(model_version=model_version, drift_type=drift_type).inc()