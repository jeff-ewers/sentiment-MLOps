from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from datetime import datetime, timedelta
from schemas.validation import PredictionRequest, PredictionResponse
from utils.logging_config import setup_logging
from database import init_db, get_db
from repositories.prediction_repository import PredictionRepository
from pydantic import ValidationError

app = Flask(__name__)
logger = setup_logging()

#initialize database
init_db()

#init model / TODO: revise for production
sentiment_analyzer = pipeline("sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f")

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


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for sentiment analysis predictions"""
    try:
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
            confidence=result[0]['score']
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
            metadata={"request_timestamp": datetime.utcnow().isoformat()}
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

if __name__ == '__main__':
    logger.info("Starting application")
    app.run(debug=True)
