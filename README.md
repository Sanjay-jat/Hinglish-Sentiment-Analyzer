# 🎭 Hinglish Sentiment Analyzer
 
**Because "यह movie बहुत अच्छी है" deserves better than Google Translate butchering it before your sentiment model even gets a chance.**
 
A sentiment analysis engine built specifically for **Hindi-English code-mixed text** — the way people *actually* type on Twitter, WhatsApp, and every Indian comment section on the internet. No pure Hindi. No pure English. Just the real, chaotic, beautiful mess of Hinglish.
 
🔗 **Live App:** [hinglish-sentiment-frontend.onrender.com](https://hinglish-sentiment-frontend.onrender.com)
🔗 **API Docs:** [hinglish-sentiment-backend.onrender.com/docs](https://hinglish-sentiment-backend.onrender.com/docs)
 
> ⏳ **Heads up:** This is hosted on Render's free tier, which puts the backend to sleep after 15 minutes of inactivity. If the app feels slow on your first try, that's just the model waking up (~30–60 seconds) — not a bug. Give it a moment and it'll be snappy after that.
 
---
 
## 🧠 What This Actually Does
 
Type a sentence like this:
 
> `"यार वो movie itni bakwaas thi, waste of time."`
 
And get back:
 
> `NEGATIVE (confidence: 94.2%)`
 
No transliteration tricks. No translating to English first and losing all the nuance. The model reads Hinglish as Hinglish.
 
---
 
## 🤔 Why This Is Harder Than It Sounds
 
Sentiment analysis is a solved problem for English. It is *not* a solved problem for Hinglish, and here's why:
 
- **No standard spelling.** "अच्छा", "acha", "achha", "accha" — all the same word, four different spellings, and a naive tokenizer treats them as four unrelated strangers.
- **Script-switching mid-sentence.** Devanagari and Latin script often show up in the *same tweet*, sometimes the *same sentence*.
- **No labeled data at scale.** Unlike English, there's no massive pretrained sentiment corpus lying around for this.
This project tackles all three head-on, using subword tokenization and pretrained embeddings specifically chosen to handle exactly this kind of linguistic chaos.
 
---
 
## 🏗️ How It's Built
 
```
Raw Tweets (SemEval-2020 Task 9 — SentiMix Hinglish)
        ↓
Text Cleaning (strip RT, mentions, whitespace noise)
        ↓
SentencePiece Tokenization (8,000-token BPE vocabulary)
        ↓
FastText Embeddings (300-dim, Hindi Common Crawl vectors)
        ↓
1D CNN — Yoon Kim architecture
   (parallel conv filters of size 2/3/4/5 → max-pool → dense → softmax)
        ↓
FastAPI backend (/predict endpoint)
        ↓
Gradio frontend (talks to backend over real HTTP, not a local import)
```
 
### Why a CNN, not a Transformer?
 
Because not every problem needs a sledgehammer. Sentiment in short, tweet-length text is usually carried by local phrase-level cues — "bahut bekaar", "ekdum mast", "bilkul sahi" — not long-range grammatical dependencies. A lightweight CNN captures this efficiently, trains in minutes instead of hours, and is cheap enough to run comfortably on a free-tier CPU server. Sometimes the boring architecture is the right architecture.
 
### Why SentencePiece + FastText?
 
- **SentencePiece** breaks words into subword pieces learned directly from the corpus, so spelling variants share structure instead of being total strangers to the model.
- **FastText** embeddings are themselves built from character n-grams, meaning even words the model has never seen get a sensible starting vector instead of a random guess.
Together, they give the model a fighting chance against a language that refuses to sit still.
 
---
 
## 📊 Results — And an Honest Confession
 
**Macro F1: 65%** on a held-out test set with zero exact overlap with training data.
 
That number might look modest next to headlines claiming 90%+ accuracy — but here's the thing: those numbers are usually a lie, and this project actually caught one in the act. During development, this model briefly hit **99% accuracy**, which is basically impossible for 3-class sentiment analysis and was a dead giveaway of data leakage. A full investigation followed: checking for duplicate tweets across train/test splits, verifying near-duplicate contamination, decoding raw tokenized input to manually sanity-check predictions, and rebuilding the pipeline from scratch once the leak was found and patched.
 
The real result — 65% macro F1 — sits comfortably alongside the official **SemEval-2020 Task 9** competition benchmarks, where top BERT-based submissions topped out around **75% F1**. A lightweight CNN landing within ~10 points of a fine-tuned transformer, on messy real-world social media text, is a legitimately solid result.
 
**Where it struggles:** the *neutral* class — which, if you think about it, makes complete sense. "Neutral" isn't the absence of sentiment words; it's sentiment-adjacent language (polite phrases, routine greetings, informational tone) that doesn't actually carry strong emotion. Telling "थैंक यू" apart from genuine enthusiasm requires contextual understanding that static embeddings and local convolutions simply don't have. It's a well-documented hard case in sentiment analysis research, not a flaw unique to this model.
 
---
 
## ⚙️ Tech Stack
 
| Layer | Tools |
|---|---|
| Tokenization | SentencePiece (BPE) |
| Embeddings | FastText (Hindi, 300-dim) |
| Model | PyTorch — 1D CNN (Yoon Kim architecture) |
| Backend | FastAPI |
| Frontend | Gradio |
| Training | Google Colab (GPU) |
| Deployment | Render (2 independent services) |
| Data | SemEval-2020 Task 9 (SentiMix Hinglish) |
 
---
 
## 🚀 Running It Locally
 
**1. Clone and install:**
```bash
git clone https://github.com/Sanjay-jat/Hinglish-Sentiment-Analyzer.git
cd Hinglish-Sentiment-Analyzer
pip install -r requirements.txt
```
 
**2. Start the backend:**
```bash
cd app
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive API.
 
**3. Start the frontend** (in a separate terminal):
```bash
cd app
python gradio_app.py
```
Visit `http://127.0.0.1:7860`.
 
---
 
## 🔌 API Reference
 
**POST** `/predict`
 
```json
// Request
{
  "text": "यह movie बहुत अच्छी है"
}
 
// Response
{
  "label": "positive",
  "confidence": 0.9142
}
```
 
---
 
## 📁 Project Structure
 
```
Hinglish-sentiment/
├── app/
│   ├── main.py          # FastAPI backend
│   ├── inference.py      # Model loading + prediction logic
│   └── gradio_app.py      # Gradio frontend
├── models/
│   ├── hinglish_cnn.pt    # Trained model weights
│   ├── hinglish_sp.model  # SentencePiece tokenizer
│   └── config.json        # Model architecture config
├── notebooks/
│   └── Hinglish_Sentiment.ipynb   # Full training pipeline
├── requirements.txt
└── README.md
```
 
---
 
## 🙋 Who This Is For
 
- **Recruiters/interviewers** who want to see the full ML lifecycle done properly: data investigation, honest evaluation, a real debugging story, and actual deployment — not just a Jupyter notebook that says "accuracy: 0.98" with no explanation.
- **Developers** working with Indian-language NLP who need a starting point for Hinglish text classification.
- **Anyone curious** what happens when you feed a neural network the exact way Indians actually type online.
---
 
## 👤 Author
 
**Sanjay Jat**
🔗 [GitHub](https://github.com/Sanjay-jat) · [LinkedIn](https://www.linkedin.com/in/sanjay-jat/)
 
---
 
*Built with genuine, hard-earned debugging scars and a healthy suspicion of any accuracy number above 90%.*