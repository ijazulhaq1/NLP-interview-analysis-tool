from flask import Flask, request, jsonify
from topic_modeling import TopicModeler
from sentiment_analysis import SentimentAnalyzer
from json_handler import JsonHandler
import os
from visualization import Visualizer
import base64
from io import BytesIO

app = Flask(__name__)
analyzer = SentimentAnalyzer()


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    analysis_type = data.get('type')

    if analysis_type == 'sentiment':
        results = {}
        for text in data['texts']:
            results[text[:20]] = analyzer.analyze(text)
        return jsonify(results)

    elif analysis_type == 'topics':
        results = TopicModeler.extract_topics(
            data['texts'],
            algorithm=data.get('algorithm', 'nmf'),
            n_topics=data.get('n_topics', 4),
            stopwords=data.get('stopwords')
        )
        TopicModeler.generate_wordcloud(results, "./static/wordclouds")
        return jsonify(results)

    return jsonify({"error": "Invalid analysis type"}), 400


if __name__ == '__main__':
    app.run(debug=True)