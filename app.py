import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, jsonify

# Download the vader_lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

sia = SentimentIntensityAnalyzer()

def sentiment_analysis(review):
    sentiment = sia.polarity_scores(review)
    # Convert the sentiment score between 0 and 5, round to 2 decimal places
    sentiment = round((sentiment['compound'] + 1) * 2.5, 2)
    return sentiment

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.get_json()['review']
        sentiment = sentiment_analysis(review)
        return jsonify({ 'sentiment': sentiment })

    return render_template('index.html')

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    data = request.get_json()
    if not data or 'review' not in data:
        return jsonify({'error': 'Bad Request', 'message': 'Request payload must include a review'}), 400

    review = data['review']
    sentiment = sentiment_analysis(review)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
