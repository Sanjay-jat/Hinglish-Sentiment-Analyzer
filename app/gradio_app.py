import gradio as gr
from inference import predict


API_URL = "http://127.0.0.1:8000/predict"


def gradio_predict(text):
    result = predict(text)
    return f"{result['label'].upper()} (confidence: {result['confidence']*100:.1f}%)"


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
    demo.launch()