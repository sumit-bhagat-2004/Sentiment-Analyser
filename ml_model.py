import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Download dataset
nltk.download("movie_reviews")

# Load data
def load_data():
    pos_reviews = [(movie_reviews.raw(fileid), "pos") for fileid in movie_reviews.fileids("pos")]
    neg_reviews = [(movie_reviews.raw(fileid), "neg") for fileid in movie_reviews.fileids("neg")]

    data = pos_reviews + neg_reviews
    texts, labels = zip(*data)
    return texts, labels

# Train the model
def train_model():
    texts, labels = load_data()

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = [1 if label == "pos" else 0 for label in labels]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save the model and vectorizer
    with open("sentiment_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
