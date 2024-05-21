import joblib

# Load the saved model and vectorizer
model = joblib.load('youtube_view_count_predictor.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict view count for a given title
def predict_view_count(title):
    X = vectorizer.transform([title]).toarray()
    prediction = model.predict(X)
    return prediction[0]

# Sample titles for testing
test_titles = [
    "How to train a neural network",
    "Top 10 programming languages in 2024",
    "Beginner's guide to machine learning",
    "Best practices for software development",
    "Advanced Python tutorials"
]

# Predict and print view counts for each title
for title in test_titles:
    predicted_count = predict_view_count(title)
    print(f"Title: {title}")
    print(f"Predicted View Count: {predicted_count}")
    print("-" * 40)
