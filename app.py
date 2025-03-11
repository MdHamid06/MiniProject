from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests

print("Starting Flask app...")

app = Flask(__name__)
print("Flask app created.")

# ✅ Load the trained model and tokenizer
try:
    model = load_model("sentiment_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model fails to load

try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None  # Prevent crashes if tokenizer fails to load

# ✅ Your CollectAPI Key for IMDb Reviews
COLLECT_API_KEY = "3DKrj6HmKvKVFqFczOmuzz:2Y2Yj6JFDfdYkHRhCHzLJg"

# ✅ Function to fetch real IMDb reviews from CollectAPI
def get_movie_reviews(movie_name):
    """Fetch real IMDb reviews from CollectAPI."""

    url = f"https://api.collectapi.com/imdb/imdbSearchByName?query={movie_name}"
    
    headers = {
        "authorization": f"apikey {COLLECT_API_KEY}",
        "content-type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching movie data: {response.json()}")
        return ["No reviews available."]

    data = response.json()

    # ✅ Extract IMDb reviews if available
    reviews = [movie["imdbContent"] for movie in data.get("result", []) if "imdbContent" in movie]

    return reviews if reviews else ["No reviews available."]

# ✅ Function to predict sentiment of a review
def predict_sentiment(review):
    if model is None or tokenizer is None:
        return "Unknown"  # Prevent crashes if model/tokenizer failed to load

    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]  # Ensure correct indexing
    return "Positive" if prediction > 0.5 else "Negative"

# ✅ Function to analyze overall sentiment of a movie
def analyze_movie_sentiment(movie_name):
    reviews = get_movie_reviews(movie_name)

    if not reviews:
        return {"error": "No reviews found for this movie!"}

    positive_count = sum(1 for review in reviews if predict_sentiment(review) == "Positive")
    total_reviews = len(reviews)
    percentage = round((positive_count / total_reviews) * 100, 2)

    overall_sentiment = "Positive" if percentage > 50 else "Negative"

    return {
        "total_reviews": total_reviews,
        "positive_percentage": percentage,
        "overall_sentiment": overall_sentiment,
        "sample_reviews": reviews[:5]  # Send 5 sample reviews in response
    }

# ✅ Home Route
@app.route("/")
def home():
    print("Home route accessed.")
    return render_template("index.html")

# ✅ Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "movie_name" not in data:
            return jsonify({"error": "No movie name provided!"}), 400

        movie_name = data["movie_name"].strip()
        result = analyze_movie_sentiment(movie_name)

        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "Something went wrong!"}), 500

# ✅ Run Flask App
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
