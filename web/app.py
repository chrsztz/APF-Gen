from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.export.txt_to_musicxml import convert_txt_to_musicxml


class ConvertRequest(BaseModel):
    txt: str
    bpm: float = 120.0
    time_signature: str = "4/4"
    min_note: str = "1/16"


app = FastAPI(title="PIG TXT to MusicXML")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "message": "POST /convert with txt, bpm, time_signature, min_note"}


@app.post("/convert")
async def convert(req: ConvertRequest):
    if not req.txt.strip():
        raise HTTPException(status_code=400, detail="txt content is empty")
    try:
        xml_str = convert_txt_to_musicxml(
            req.txt,
            bpm=req.bpm,
            time_signature=req.time_signature,
            min_note=req.min_note,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"xml": xml_str}

