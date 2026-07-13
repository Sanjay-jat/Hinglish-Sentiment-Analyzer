# Hinglish Sentiment Analyzer
 
> Sentiment analysis for Hindi-English code-mixed text — the way India actually types. Custom CNN + FastText + SentencePiece pipeline, trained from scratch on SemEval-2020 Task 9, deployed as a live two-service app.
 
![Macro F1](https://img.shields.io/badge/Macro%20F1-65%25-yellow?style=for-the-badge) ![Vocab](https://img.shields.io/badge/Vocab-8K%20BPE-blue?style=for-the-badge) ![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red?style=for-the-badge&logo=pytorch) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=for-the-badge&logo=fastapi) ![Gradio](https://img.shields.io/badge/Gradio-frontend-orange?style=for-the-badge) ![deployed](https://img.shields.io/badge/deployed-live-brightgreen?style=for-the-badge)
 
### 🚀 [Live Demo](https://hinglish-sentiment-frontend.onrender.com) &nbsp;·&nbsp; 🔌 [API Docs](https://hinglish-sentiment-backend.onrender.com/docs)
 
Deployed on **Render** — two independent services (backend + frontend) talking over real HTTP.
 
> ⏳ **Free-tier note:** The backend sleeps after 15 min idle. First request after a nap takes ~30–60s to wake it up — after that, it's instant. Not a bug, just the price of "free forever."
 
---
 
## Screenshots
 
### Live App — Sentiment in Real Time
 
Type Hinglish. Get a label and a confidence score. That's it.
 
*(Add a screenshot of the running app here — text box, prediction, examples)*
 
---
 
## What This Actually Does
 
Type a sentence like:
 
```
यार वो movie itni bakwaas thi, waste of time.
```
 
Get back:
 
```
NEGATIVE (confidence: 94.2%)
```
 
No transliteration hacks. No translate-to-English-first-and-lose-the-nuance shortcuts. The model reads Hinglish as Hinglish — because that's how ~600 million people online actually write.
 
---
 
## Why This Is Harder Than It Looks
 
Sentiment analysis on English is a solved problem. Hinglish is not:
 
- **No standard spelling** — "अच्छा" / "acha" / "achha" / "accha" are the same word, but a naive tokenizer sees four strangers.
- **Script-switching mid-sentence** — Devanagari and Latin routinely show up in the *same tweet*.
- **No pretrained sentiment corpus at scale** — unlike English, there's no giant labeled dataset to lean on.
This project handles all three with subword tokenization and pretrained embeddings picked specifically for this chaos.
 
---
 
## Tech Stack
 
<table>
<tr>
<td align="center" width="140"><img src="https://img.shields.io/badge/-SentencePiece-4B8BBE?style=for-the-badge" /><br><sub>Tokenizer</sub></td>
<td align="center" width="140"><img src="https://img.shields.io/badge/-FastText-1877F2?style=for-the-badge" /><br><sub>Embeddings</sub></td>
<td align="center" width="140"><img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" /><br><sub>Model (CNN)</sub></td>
<td align="center" width="140"><img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" /><br><sub>Backend</sub></td>
</tr>
<tr>
<td align="center"><img src="https://img.shields.io/badge/-Gradio-FF7C00?style=for-the-badge" /><br><sub>Frontend</sub></td>
<td align="center"><img src="https://img.shields.io/badge/-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" /><br><sub>Training</sub></td>
<td align="center"><img src="https://img.shields.io/badge/-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white" /><br><sub>Deployment</sub></td>
<td align="center"><img src="https://img.shields.io/badge/-SemEval%202020-6f42c1?style=for-the-badge" /><br><sub>Dataset</sub></td>
</tr>
</table>
---
 
## Architecture
 
```
Raw Tweets (SemEval-2020 Task 9)
        ↓
Text Cleaning (strip RT, mentions, whitespace noise)
        ↓
SentencePiece Tokenization (8K BPE vocab)
        ↓
FastText Embeddings (300-dim, Hindi Common Crawl)
        ↓
1D CNN — parallel conv filters (size 2/3/4/5) → max-pool → dense → softmax
        ↓
FastAPI (/predict) ←→ Gradio (real HTTP, not a local import)
```
 
**Why a CNN, not a transformer?** Sentiment in tweet-length text is usually carried by local phrase cues — "bahut bekaar", "ekdum mast" — not long-range grammar. A CNN captures this efficiently, trains in minutes, and runs comfortably on a free CPU server.
 
**Why SentencePiece + FastText?** SentencePiece splits words into subword pieces learned from the corpus, so spelling variants share structure instead of being strangers. FastText embeddings are built from character n-grams, so even unseen words get a sensible starting vector.
 
---
 
## Results
 
| Metric | Score |
|---|---|
| Train Accuracy | 95.2% |
| Validation Macro F1 | 99.4% *(same split, near-memorization — see note)* |
| **Test Macro F1 (clean, held-out)** | **65.6%** |
| Test Accuracy (clean, held-out) | 65.7% |
| SemEval-2020 top BERT benchmark | ~75% F1 |
 
**The 65% is the real number.** Mid-training this model briefly scored 99% — impossible for 3-class sentiment. Turned out to be data leakage (duplicate tweets across train/test splits in the public dataset mirror). Found it, fixed it, rebuilt the pipeline clean, and landed within ~10 points of a fine-tuned BERT baseline using a CNN that trains in minutes. **Weakest class: neutral** — it's sentiment-*adjacent* language (polite phrases, greetings) without real emotion, which needs more context than a CNN captures.
 
---
 
## Running It Locally
 
```bash
git clone https://github.com/Sanjay-jat/Hinglish-Sentiment-Analyzer.git
cd Hinglish-Sentiment-Analyzer
pip install -r requirements.txt
```
 
**Backend:**
```bash
cd app
uvicorn main:app --reload
# → http://127.0.0.1:8000/docs
```
 
**Frontend** (separate terminal):
```bash
cd app
python gradio_app.py
# → http://127.0.0.1:7860
```
 
---
 
## API Reference
 
**POST** `/predict`
 
```json
// Request
{ "text": "यह movie बहुत अच्छी है" }
 
// Response
{ "label": "positive", "confidence": 0.9142 }
```
 
---
 
## Project Structure
 
```
Hinglish-sentiment/
├── app/
│   ├── main.py            # FastAPI backend
│   ├── inference.py        # Model loading + prediction logic
│   └── gradio_app.py        # Gradio frontend
├── models/
│   ├── hinglish_cnn.pt      # Trained model weights
│   ├── hinglish_sp.model    # SentencePiece tokenizer
│   └── config.json          # Model architecture config
├── notebooks/
│   └── Hinglish_Sentiment.ipynb
├── requirements.txt
└── README.md
```
 
---
 
## About Me
 
**Sanjay Jat**
3rd-year BTech CSE student building an AI/Backend engineering portfolio, one honestly-debugged project at a time.
 
[![GitHub](https://img.shields.io/badge/GitHub-Sanjay--jat-181717?style=for-the-badge&logo=github)](https://github.com/Sanjay-jat) [![LinkedIn](https://img.shields.io/badge/LinkedIn-sanjay--jat-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sanjay-jat/)
 
