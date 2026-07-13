# Hinglish Sentiment Analyzer
 
> Sentiment analysis for Hindi-English code-mixed text — the way India actually types. Custom CNN + FastText + SentencePiece pipeline, trained from scratch on SemEval-2020 Task 9, deployed as a live two-service app.
 
![Macro F1](https://img.shields.io/badge/Macro%20F1-65%25-yellow) ![Vocab](https://img.shields.io/badge/Vocab-8K%20BPE-blue) ![FastText Coverage](https://img.shields.io/badge/FastText%20Coverage-86.8%25-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-CNN-red) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688) ![Gradio](https://img.shields.io/badge/Gradio-frontend-orange) ![deployed](https://img.shields.io/badge/deployed-live-brightgreen)
 
🚀 **Live Demo →** [hinglish-sentiment-frontend.onrender.com](https://hinglish-sentiment-frontend.onrender.com)
🔌 **API Docs →** [hinglish-sentiment-backend.onrender.com/docs](https://hinglish-sentiment-backend.onrender.com/docs)
 
Deployed on **Render** (backend) · **Render** (frontend) — two independent services talking over real HTTP.
 
> ⏳ **Free-tier note:** The backend sleeps after 15 min of inactivity. First request after a nap takes ~30–60s to wake it up — after that, it's instant. Not a bug, just the price of "free forever."
 
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
 
No transliteration hacks. No translate-to-English-first-and-lose-all-the-nuance shortcuts. The model reads Hinglish as Hinglish — because that's how ~600 million people online actually write.
 
---
 
## Why This Is Harder Than It Looks
 
Sentiment analysis on English is a solved problem. Hinglish is a different beast entirely:
 
- **No standard spelling.** "अच्छा" / "acha" / "achha" / "accha" are the same word — a naive tokenizer sees four unrelated strangers.
- **Script-switching mid-sentence.** Devanagari and Latin script routinely show up in the *same tweet*.
- **No pretrained sentiment corpus at scale.** Unlike English, there's no giant labeled dataset lying around.
This project handles all three with subword tokenization and pretrained embeddings chosen specifically for this kind of chaos — not bolted on as an afterthought.
 
---
 
## Tech Stack
 
| Layer | Choice |
|---|---|
| 🔤 Tokenization | SentencePiece — 8,000-token BPE vocabulary |
| 🧬 Embeddings | FastText (Hindi, 300-dim, Common Crawl) |
| 🧠 Model | PyTorch — 1D CNN, Yoon Kim architecture |
| ⚡ Backend | FastAPI |
| 🎛️ Frontend | Gradio |
| 🏋️ Training | Google Colab (GPU) |
| ☁️ Deployment | Render — two independent services |
| 📚 Dataset | SemEval-2020 Task 9 (SentiMix Hinglish) |
 
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
 
**Why a CNN, not a transformer?** Sentiment in short, tweet-length text is usually carried by local phrase cues — "bahut bekaar", "ekdum mast", "bilkul sahi" — not long-range grammar. A CNN captures this efficiently, trains in minutes, and runs comfortably on a free CPU server. The boring architecture is sometimes the right one.
 
**Why SentencePiece + FastText?** SentencePiece breaks words into subword pieces learned from the corpus, so spelling variants share structure instead of being strangers. FastText embeddings are themselves built from character n-grams, so even unseen words get a sensible starting vector instead of a random guess.
 
---
 
## Results — And a Confession
 
**Macro F1: 65%** on a held-out test set with zero exact overlap with training data.
 
That number looks modest next to headline claims of 90%+ accuracy — and that's exactly the point. Mid-development, this model briefly hit **99% accuracy**, which is essentially impossible for 3-class sentiment analysis. Instead of shipping it, the number got investigated: checked for duplicate tweets across splits, verified near-duplicate contamination separately, manually decoded raw tokenized input to sanity-check predictions by hand, and rebuilt the pipeline from scratch once the leak was confirmed and patched.
 
The real result — 65% macro F1 — sits right alongside the official **SemEval-2020 Task 9** benchmarks, where top BERT-based submissions topped out around **75% F1**. A lightweight CNN landing within ~10 points of a fine-tuned transformer, on real, messy social-media text, is a legitimately solid result.
 
**Weakest class: neutral.** Makes sense once you think about it — "neutral" isn't the *absence* of sentiment words, it's sentiment-adjacent language (polite phrases, routine greetings) that doesn't actually carry strong emotion. Separating "थैंक यू" from genuine enthusiasm needs contextual understanding that static embeddings and local convolutions don't have. A well-documented hard case in the literature, not a flaw unique to this build.
 
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
 
[![GitHub](https://img.shields.io/badge/GitHub-Sanjay--jat-181717?logo=github)](https://github.com/Sanjay-jat) [![LinkedIn](https://img.shields.io/badge/LinkedIn-sanjay--jat-0077B5?logo=linkedin)](https://www.linkedin.com/in/sanjay-jat/)
 
*If your accuracy is above 95% on a 3-class problem, check for data leakage before you check for a job.*
