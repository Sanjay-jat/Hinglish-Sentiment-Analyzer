from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict

app = FastAPI(title="Hinglish Sentiment Analyzer")

class PredictionRequest(BaseModel):
    text:str

class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.get('/')
def root():
    return {"message":"Hinglish Sentiment Analyzer API is running"}

@app.post("/predict", response_model=PredictionResponse)
def prediction(request:PredictionRequest):
    result=predict(request.text)
    return result