from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from datetime import datetime, timedelta
from schemas.validation import PredictionRequest, PredictionResponse
from utils.logging_config import setup_logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.database.models import init_db, get_db
from app.repositories.prediction_repository import PredictionRepository
from app.monitoring.drift import BayesianDriftDetector
from app.schemas.monitoring import DriftDetectionRequest, ChangePointResponse
from app.repositories.drift_repository import DriftRepository
from app.monitoring.metrics import MetricsTracker
import time
import random
from pydantic import ValidationError

MODEL_CONFIG = {
    "version": "v1",
    "name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "revision": "714eb0f"
}

app = Flask(__name__)
logger = setup_logging()

#initialize database
init_db()

#init model
# TODO: revise for production
sentiment_analyzer = pipeline("sentiment-analysis",
    model=MODEL_CONFIG["name"],
    revision=MODEL_CONFIG["revision"])

#env test endpoints

@app.route('/health', methods=['GET'])
def health_check():
    status = {
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'dependencies_loaded': True
    }
    logger.info("Health check", extra={'status': status})
    return jsonify(status)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint for exposing Prometheus metrics"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for sentiment analysis predictions"""
    try:
        start_time = time.time()
        try:
            #validate request data
            request_data = PredictionRequest(**request.json)
        except ValidationError as e:
            logger.error("validation error", extra={'errors': e.errors()})
            return jsonify({'error': 'Validation failed', 'details': e.errors()}), 400
        
        #log incoming request
        logger.info("Received prediction request", extra={'text': request_data.text})

        #run inference
        result = sentiment_analyzer(request_data.text)

        response = PredictionResponse(
            text=request_data.text,
            sentiment=result[0]['label'],
            confidence=result[0]['score'],
            model_version=MODEL_CONFIG['version']
        )

        #track metrics
        latency = time.time() - start_time
        MetricsTracker.track_prediction(
            model_version=response.model_version,
            sentiment=response.sentiment,
            confidence=response.confidence,
            latency=latency
        )

        #store prediction in database
        db = next(get_db())
        repo = PredictionRepository(db)
        repo.create_prediction(
            text=response.text,
            sentiment=response.sentiment,
            confidence=response.confidence,
            model_version=response.model_version,
            raw_model_output=result[0],
            request_metadata={"request_timestamp": datetime.utcnow().isoformat()}
        )

        #log prediction
        logger.info("Generated prediction", extra={'prediction': response.dict()})

        return jsonify(response.dict())

    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({'error': str(e)}), 500
    
@app.route('/predictions/recent', methods=['GET'])
def get_recent_predictions():
    """Get recent prediction history"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        db = next(get_db())
        repo = PredictionRepository(db)
        predictions = repo.get_recent_predictions(limit=limit)

        return jsonify([{
            'text': p.text,
            'sentiment': p.sentiment,
            'confidence': p.confidence,
            'model_version': p.model_version,
            'created_at': p.created_at.isoformat()
        } for p in predictions])
    
    except Exception as e:
        logger.exception("Error fetching recent predictions")
        return jsonify({'error': str(e)}), 500
    
@app.route('/predictions/stats', methods=['GET'])
def get_prediction_stats():
    """Get prediction statistics"""
    try:
        hours = request.args.get('hours', default=24, type=int)
        model_version = request.args.get('model_version', default=None, type=str)
    
        db = next(get_db())
        repo = PredictionRepository(db)
        stats = repo.get_model_performance_stats(
            model_version=model_version,
            timeframe_hours=hours
        )

        return jsonify(stats)
    
    except Exception as e:
        logger.exception("Error fetching prediction stats")
        return jsonify({'error': str(e)}), 500
    
@app.route('/monitoring/drift/detect', methods=['POST'])
def detect_drift():
    """Endpoint to run drift detection analysis"""
    try:
        request_data = DriftDetectionRequest(**request.json)

        #get predictions for analysis
        db = next(get_db())
        pred_repo = PredictionRepository(db)
        drift_repo = DriftRepository(db)

        #configure detector
        detector = BayesianDriftDetector(
            window_size=timedelta(hours=request_data.window_size_hours),
            min_probability=request_data.min_probability
        )

        #get predictions for analysis window
        analysis_window = request_data.window_size_hours * 2 # x2 for before/after
        predictions = pred_repo.get_predictions_by_timeframe(
            start_time=datetime.utcnow() - timedelta(hours=analysis_window),
            end_time=datetime.utcnow()
        )

        #run detection
        sentiment_changes = detector.detect_sentiment_drift(predictions)
        confidence_changes = detector.detect_confidence_drift(predictions)

        #track drift detection metrics
        drift_types = {}
        if sentiment_changes:
            drift_types['sentiment'] = len(sentiment_changes)
        if confidence_changes:
            drift_types['confidence'] = len(confidence_changes)

        MetricsTracker.track_drift_detection(
            model_version=request_data.model_version,
            detected_drifts=len(sentiment_changes) + len(confidence_changes),
            drift_types=drift_types
        )

        #save results
        all_changes = []
        for change in sentiment_changes + confidence_changes:
            drift_repo.save_changepoint(
                timestamp=change.timestamp,
                confidence=change.confidence,
                metric=change.metric,
                before_params=change.before_params,
                after_params=change.after_params,
                model_version=request_data.model_version
            )
            all_changes.append(ChangePointResponse(**change.__dict__))
        
        return jsonify([change.dict() for change in all_changes])
    
    except Exception as e:
        logger.exception("Error in drift detection")
        return jsonify({'error': str(e)}), 500
    
@app.route('/monitoring/drift/history', methods=['GET'])
def get_drift_history():
    """Get historical drift detection results"""
    try:
        model_version = request.args.get('model_version')

        if not model_version:
            return jsonify({'error': 'Model version is required'}), 400

        hours = request.args.get('hours', default=168, type=int)
        metric = request.args.get('metric', default=None, type=str)
        

        db = next(get_db())
        drift_repo = DriftRepository(db)

        changepoints = drift_repo.get_recent_changepoints(
            hours=hours,
            metric=metric,
            model_version=model_version
        )

        return jsonify([{
            'timestamp': cp.timestamp.isoformat(),
            'confidence': cp.confidence,
            'metric': cp.metric,
            'before_params': cp.before_params,
            'after_params': cp.after_params,
            'model_version': cp.model_version
        } for cp in changepoints])

    except Exception as e:
        logger.exception("Error fetching drift history")
        return jsonify({'error': str(e)}), 500

@app.route('/test/generate-data', methods=['POST'])
def generate_test_data():
    """Generate test data with known drift patterns"""
    try:
        db = next(get_db())
        repo = PredictionRepository(db)

        #generate data over past week
        base_time = datetime.utcnow() - timedelta(days=7)

        #first 3 days - mostly POSITIVE
        for i in range(100):
            time = base_time + timedelta(
                days=random.uniform(0,3),
                hours=random.uniform(0,24)
            )
            repo.create_prediction(
                text=f"Sample text {i}",
                sentiment="POSITIVE" if random.random() < 0.8 else "NEGATIVE",
                confidence=random.uniform(0.8, 0.99),
                model_version="v1",
                raw_model_output={"score": 0.9},
                request_metadata={"timestamp": time.isoformat()}
            )
        #next 4 days - SIMULATED DRIFT
        for i in range(100):
            time = base_time + timedelta(
                days=random.uniform(3,7),
                hours=random.uniform(0,24)
            )
            repo.create_prediction(
                text=f"Sample text {i}",
                sentiment="NEGATIVE" if random.random() < 0.8 else "POSITIVE",
                confidence=random.uniform(0.8, 0.99),
                model_version="v1",
                raw_model_output={"score": 0.9},
                request_metadata={"timestamp": time.isoformat()}
            )
        
        return jsonify("message": "Generated 200 test predictions with drift pattern")
    
    except Exception as e:
        logger.exception("Error generating test data")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting application")
    app.run(debug=True, port=5001)
