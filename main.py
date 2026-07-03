from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="ML Sentiment API")

# Train ML model with sample data
texts = ["I love this", "This is terrible", "Amazing product", "Worst experience", "Great job"]
labels = [1, 0, 1, 0, 1] # 1=Positive, 0=Negative

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ML Sentiment API is running. Go to /docs to test"}

@app.post("/predict")
def predict(data: TextInput):
    text_vector = vectorizer.transform([data.text])
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return {"text": data.text, "sentiment": sentiment}