from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {
        "result": response.output_text
    }
