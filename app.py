from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Load trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"

    response = jsonify({"sentiment": sentiment})
    
    # Add CORS headers
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")

    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
