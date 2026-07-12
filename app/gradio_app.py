import gradio as gr
from inference import predict
import os

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")


def gradio_predict(text):
    response = requests.post(API_URL, json={"text": text})
    if response.status_code == 200:
        result = response.json()
        return f"{result['label'].upper()} (confidence: {result['confidence']*100:.1f}%)"
    else:
        return f"Error: {response.status_code} - {response.text}"


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(label="Enter Hinglish text", placeholder="यह movie बहुत अच्छी है..."),
    outputs=gr.Textbox(label="Sentiment Prediction"),
    title="Hinglish Sentiment Analyzer",
    description="Enter Hindi-English code-mixed text (like tweets) to detect sentiment — negative, neutral, or positive. ""If a prediction looks wrong, click 'Flag' below to save it for review.",
    examples=[
        ["यह movie बहुत अच्छी है"],
        ["worst movie है ये"],
        ["आज weather ठीक है"]
    ],
    flagging_mode="manual",
    flagging_options=["Wrong prediction"]
)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)