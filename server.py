import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.allowed_locations = [
            "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
            "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
            "El Paso, Texas", "Escondido, California", "Fresno, California",
            "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
            "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
            "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
        ]

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query = environ.get('QUERY_STRING', '')
            params = parse_qs(query)
            location = params.get('location', [None])[0]
            start_date = params.get('start_date', [None])[0]
            end_date = params.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location and location in self.allowed_locations:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment

            sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(sorted_reviews).encode('utf-8')

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
            params = parse_qs(request_body)
            review_body = params.get('ReviewBody', [None])[0]
            location = params.get('Location', [None])[0]

            if not review_body or not location:
                response_body = json.dumps({"error": "ReviewBody and Location are required"}).encode('utf-8')
                start_response('400 BAD REQUEST', [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            if location not in self.allowed_locations:
                response_body = json.dumps({"error": "Invalid location"}).encode('utf-8')
                start_response('400 BAD REQUEST', [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp
            }

            reviews.append(new_review)

            response_body = json.dumps(new_review).encode('utf-8')
            start_response('201 CREATED', [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()