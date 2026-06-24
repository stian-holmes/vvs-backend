import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VVS-Rapport API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=api_key, timeout=30.0)


class InputData(BaseModel):
    description: str


class ReportResponse(BaseModel):
    result: str


@app.get("/")
async def root():
    """Root endpoint - confirms API is running."""
    logger.info("Root endpoint called")
    return {"message": "API fungerer ✅"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/generate", response_model=ReportResponse)
async def generate(data: InputData):
    """Generate a VVS report based on the provided description."""
    
    # Validate input
    if not data.description.strip():
        logger.warning("Empty description provided")
        raise HTTPException(
            status_code=400,
            detail="Description cannot be empty"
        )

    logger.info(f"Generating report for description: {data.description[:50]}...")

prompt = f"""
Du er en erfaren rørlegger.

Lag en profesjonell og kundeklar VVS-rapport basert på dette:

{data.description}

Krav:
- Skriv kort, tydelig og profesjonelt språk
- Ikke bruk AI-aktige formuleringer
- Skriv som en rapport til kunde

Struktur:

VVS-Rapport

Dato:
Kunde:
Adresse:
Prosjekt:

Kort oppsummering:
(Situasjon forklart enkelt for kunde)

Mulig årsak:
(Forklart på en forståelig måte)

Anbefalte tiltak:
(Konkrete forslag til utbedring)

Estimert tidsforbruk:
(Bruk det som er oppgitt, og kommenter kort)

Konklusjon:
(Hva bør gjøres videre - tydelig anbefaling)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Du er en erfaren rørlegger som lager profesjonelle rapporter."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            timeout=30.0
        )

        # Validate response
        if not response.choices or not response.choices[0].message.content:
            logger.error("Invalid response from OpenAI API")
            raise HTTPException(
                status_code=500,
                detail="No valid response received from AI service"
            )

        result = response.choices[0].message.content
        logger.info("Report generated successfully")
        
        return ReportResponse(result=result)

    except RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        raise HTTPException(
            status_code=429,
            detail="API rate limit exceeded. Please try again later."
        )
    
    except APIConnectionError as e:
        logger.error(f"Connection error to OpenAI: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to AI service. Please try again later."
        )
    
    except APIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"AI service error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )
