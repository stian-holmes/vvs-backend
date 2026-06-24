from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os

# Sett API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Input modell
class InputData(BaseModel):
    description: str

@app.get("/")
def root():
    return {"message": "API fungerer ✅"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/generate")
def generate(data: InputData):
    prompt = f"""
Du er en erfaren rørlegger.
Lag en profesjonell VVS-rapport basert på dette:

{data.description}

Inkluder:
- Kort oppsummering
- Mulig årsak
- Anbefalte tiltak
- Konklusjon
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Du er en profesjonell rørlegger."},
            {"role": "user", "content": prompt}
        ]
    )

    return {
        "result": response["choices"][0]["message"]["content"]
    }
``
