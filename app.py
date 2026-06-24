from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du er en profesjonell rørlegger."},
                {"role": "user", "content": prompt}
            ]
        )

        return {
            "result": response.choices[0].message.content
        }

    except Exception as e:
        return {"error": str(e)}  # 👈 viktig: vis feilen direkte
