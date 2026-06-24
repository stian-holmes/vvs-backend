from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

# Hent API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key)


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

    if not data.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    # ✅ IMPORTANT: dette MÅ være inne i funksjonen
    prompt = f"""
Du er en erfaren rørlegger.

Lag en profesjonell og tydelig VVS-rapport basert på dette:

{data.description}

Format:
- Skriv kort og presist
- Bruk overskrifter uten #
- Struktur:

VVS-Rapport

Dato:
Kunde:
Adresse:
Prosjekt:

Kort oppsummering:
...

Mulig årsak:
...

Anbefalte tiltak:
...

Konklusjon:
...
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er en erfaren rørlegger."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return {
            "result": response.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
