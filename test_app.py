from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

#env test endpoints

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'dependencies_loaded': True
    })

@app.route('/test-model', methods=['POST'])
def test_model():
    try:
        #init model
        sentiment_analyzer = pipeline("sentiment-analysis")

        #get text from request or default
        data = request.json or {}
        text = data.get('text', 'This is a test sentence.')

        #run inference
        result = sentiment_analyzer(text)

        return jsonify({
            'text': text,
            'result': result[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
