import numpy as np
import emoji
import re
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack, csr_matrix

# Load models
with open("../model/sarcasm_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
word_vec = bundle["word_vec"]
char_vec = bundle["char_vec"]

analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cap_features(texts):

    features = []

    for text in texts:

        total = len(text)
        upper = sum(1 for c in text if c.isupper())

        ratio = upper / total if total > 0 else 0

        has_caps = any(word.isupper() and len(word) > 2 for word in text.split())

        features.append([ratio, int(has_caps)])

    return np.array(features)


def extract_emojis(text):
    return [c for c in str(text) if c in emoji.EMOJI_DATA]


def remove_emojis(text):
    return ''.join(c for c in str(text) if c not in emoji.EMOJI_DATA)


emoji_sentiment = {

    "🚀":0.95,
    "📈":0.85,
    "💰":0.8,
    "💎":0.75,
    "🐂":0.85,
    "🔥":0.7,

    "📉":-0.9,
    "🐻":-0.85,
    "💀":-0.8,
    "😭":-0.7,

    "😂":-0.2,
    "🤣":-0.2,
    "🙄":-0.3
}


def emoji_score(emojis):

    if not emojis:
        return 0

    scores = [emoji_sentiment.get(e,0) for e in emojis]

    intensity = 1 + 0.1*(len(emojis)-1)

    return np.mean(scores)*intensity


def sarcasm_probability(text):

    text = clean_text(text)

    word_part = word_vec.transform([text])
    char_part = char_vec.transform([text])
    caps_part = csr_matrix(cap_features([text]))

    combined = hstack([word_part,char_part,caps_part])

    prob = model.predict_proba(combined)[0][1]

    return float(prob)


def baseline_vader(text):

    return analyzer.polarity_scores(text)["compound"]


def emoji_vader(text):

    emojis = extract_emojis(text)
    text_clean = remove_emojis(text)

    s_text = analyzer.polarity_scores(text_clean)["compound"]
    s_emoji = emoji_score(emojis)

    return float(0.7*s_text + 0.3*s_emoji)


def es_vader(text, alpha=0.7, beta=0.3, lambda_sarc=0.8):

    emojis = extract_emojis(text)
    text_clean = remove_emojis(text)

    s_text = analyzer.polarity_scores(text_clean)["compound"]
    s_emoji = emoji_score(emojis)

    # base emoji-aware sentiment
    s_base = alpha * s_text + beta * s_emoji

    # sarcasm probability
    s_prob = sarcasm_probability(text)

    # -----------------------------
    # CONTRADICTION DETECTION
    # -----------------------------
    contradiction = (s_text > 0.4 and s_emoji < -0.1)

    # -----------------------------
    # FLIP RULE
    # -----------------------------
    if contradiction and s_prob > 0.3:
        s_base = (-alpha * s_text) + beta * s_emoji

    # -----------------------------
    # NORMAL SARCASM DAMPENING
    # -----------------------------
    adjusted = max(0, s_prob - 0.6)

    s_final = s_base * (1 - lambda_sarc * adjusted)

    return max(min(s_final,1),-1)