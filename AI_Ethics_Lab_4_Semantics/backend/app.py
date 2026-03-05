from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentiment_engine import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str


@app.post("/analyze")
def analyze(input: TextInput):

    text = input.text

    sarcasm = sarcasm_probability(text)
    vader = baseline_vader(text)
    emoji_v = emoji_vader(text)
    es = es_vader(text)

    return {
        "text": text,
        "sarcasm_probability": sarcasm,
        "vader": vader,
        "emoji_vader": emoji_v,
        "es_vader": es
    }