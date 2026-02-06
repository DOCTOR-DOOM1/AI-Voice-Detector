from fastapi import FastAPI, HTTPException, Security, Depends, File, UploadFile
import base64
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn
from inference import detector
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(
    title="AI Voice Detector API",
    description="API for distinguishing between human and AI-generated speech.",
    version="1.0.0"
)

# Security
API_KEY_NAME = "X-API-Key"
API_KEY = "hackathon-secret-key" # In production, use environment variables

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )

# Input Schema
class VoiceRequest(BaseModel):
    audio_base64: str

# Endpoints
@app.get("/")
def read_root():
    import os
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(file_path)

@app.get("/health")
def health_check():
    return {"status": "active", "model": "DualPathDA_LFCC"}

@app.post("/detect", dependencies=[Depends(get_api_key)])
async def detect_voice(request: VoiceRequest):
    """
    Analyzes the provided Base64 encoded MP3 audio.
    Returns classification (HUMAN/AI_GENERATED), confidence score, and explanation.
    """
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Missing audio_base64 field")
    
    result = detector.predict(request.audio_base64)
    
    if result["classification"] == "ERROR":
        raise HTTPException(status_code=500, detail=result["explanation"])
        
    return result

@app.post("/detect/audio-file", dependencies=[Depends(get_api_key)])
async def detect_voice_file(file: UploadFile = File(...)):
    """
    Upload an MP3/WAV file directly for analysis.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Convert to base64 to reuse the existing inference logic
        # (API was originally designed for JSON/Base64)
        b64_string = base64.b64encode(content).decode('utf-8')
        
        result = detector.predict(b64_string)
        
        if result["classification"] == "ERROR":
            raise HTTPException(status_code=500, detail=result["explanation"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
