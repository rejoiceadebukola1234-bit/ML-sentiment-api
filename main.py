from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI(title="Sentiment API")

class TextIn(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ML Sentiment API is Live!"}

@app.post("/predict")
def predict(data: TextIn):
    analysis = TextBlob(data.text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {"text": data.text, "sentiment": sentiment, "score": round(polarity, 3)}