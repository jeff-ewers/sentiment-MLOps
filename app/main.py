from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from schemas.validation import PredictionRequest, PredictionResponse
from utils.logging_config import setup_logging
from pydantic import ValidationError

app = Flask(__name__)
logger = setup_logging()

#init model / TODO: revise for production
sentiment_analyzer = pipeline("sentiment-analysis")

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


@app.route('/test-model', methods=['POST'])
def test_model():
    try:
        try:
            request_data = PredictionRequest(**request.json)
        except ValidationError as e:
            logger.error("Validation error", extra={'errors': e.errors()})
            return jsonify({'error': 'Validation failed', 'details': e.errors()}), 400
        
        #log incoming request
        logger.info("Received prediction request", extra={'text': request_data.text})


        #run inference
        result = sentiment_analyzer(text)

        response = PredictionResponse(
            text=request_data.text,
            sentiment=result[0]['label'],
            confidence=result[0]['score']
        )

        #log prediction
        logger.info("Generated prediction", extra={'prediction': response.dict()})

        return jsonify(response.dict())
            
            

    except Exception as e:
        logger.exception("Error processing request")
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    logger.info("Starting application")
    app.run(debug=True)
