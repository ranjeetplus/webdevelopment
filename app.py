from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import pipeline

from flask import Flask, send_from_directory

app = Flask(__name__)

# Load a pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Perform sentiment analysis
    result = sentiment_analyzer(text)[0]
    return jsonify({
        "text": text,
        "sentiment": result['label'],
        "confidence": result['score']
    })

if __name__ == '__main__':
    app.run(debug=True)
