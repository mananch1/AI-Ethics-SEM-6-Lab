# Bitcoin Tweet Sarcasm & Sentiment Analyzer

A hybrid NLP system that analyzes **Bitcoin-related tweets** for sentiment and sarcasm using a combination of **VADER sentiment analysis, emoji sentiment modeling, and a machine-learning sarcasm classifier**.

The project provides an interactive interface where users can input tweets and obtain multiple sentiment scores along with sarcasm probability.

---

## Features

* **Sarcasm Detection Model**

  * Trained using TF-IDF word and character features
  * Logistic Regression classifier
  * Detects sarcastic tone in social media text

* **Sentiment Analysis**

  * **VADER** baseline sentiment
  * **Emoji-aware sentiment** scoring
  * **ES-VADER**: sarcasm-adjusted sentiment

* **Emoji Sentiment Integration**

  * Domain-specific emoji lexicon for crypto discussions
  * Emoji intensity weighting
  * Sentiment contradiction detection

* **Interactive Web Interface**

  * Input tweet text
  * View sarcasm probability
  * Compare VADER, Emoji-VADER, and ES-VADER scores
  * Clickable example tweets

---

## Project Structure

```
btc_sentiment_app/
│
├── backend/
│   ├── app.py
│   └── sentiment_engine.py
│
├── frontend/
│   └── index.html
│
├── model/
│   └── sarcasm_bundle.pkl
│
├── notebook/
│   └── training_notebook.ipynb
│
└── requirements.txt
```

---

## Datasets

* Bitcoin Tweets Dataset
  https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets

* Reddit Sarcasm Dataset
  https://www.kaggle.com/datasets/danofer/sarcasm

---

## Installation

1. Clone the repository

```
git clone <repo-url>
cd btc_sentiment_app
```

2. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

Start the backend server:

```
cd backend
uvicorn app:app --reload
```

Then open the frontend:

```
frontend/index.html
```

Enter a tweet and click **Analyze Tweet** to view the sentiment and sarcasm scores.

---

## Example Input

```
Amazing investment opportunity 📉😂
```

Example output includes:

* Sarcasm Probability
* VADER Sentiment
* Emoji-VADER Sentiment
* ES-VADER (Sarcasm Adjusted Sentiment)

---

## Technologies Used

* Python
* FastAPI
* Scikit-learn
* VADER Sentiment
* NumPy / Pandas
* HTML / JavaScript

---

## Author

Manan Chahal
