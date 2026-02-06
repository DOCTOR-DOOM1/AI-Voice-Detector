from fastapi import FastAPI, HTTPException, Security, Depends, File, UploadFile
import base64
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn
from inference import detector
from fastapi.responses import HTMLResponse

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

# --- EMBEDDED FRONTEND (To fix Cloud Deployment issues) ---
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Detector | Premium</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4338ca;
            --secondary: #ec4899;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --success: #10b981;
            --danger: #ef4444;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', system-ui, -apple-system, sans-serif; }

        body {
            background-color: var(--bg);
            background-image:
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.15) 0px, transparent 50%);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container { width: 100%; max-width: 500px; padding: 2rem; }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            text-align: center;
        }

        h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        p.subtitle { color: var(--text-muted); margin-bottom: 2rem; font-size: 0.95rem; }

        .controls { margin-bottom: 2rem; text-align: left; }

        label { display: block; color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 500; }

        select {
            width: 100%;
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text);
            padding: 10px 14px;
            border-radius: 8px;
            outline: none;
            transition: all 0.2s;
        }

        select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2); }

        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 3rem 1rem;
            cursor: pointer;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }

        .upload-area:hover { border-color: var(--primary); background: rgba(99, 102, 241, 0.05); }

        .upload-icon { font-size: 2.5rem; margin-bottom: 1rem; display: block; }

        #fileName { font-size: 0.9rem; color: var(--text-muted); word-break: break-all; }

        button.cta {
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border: none;
            padding: 14px 24px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            width: 100%;
            margin-top: 1.5rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        button.cta:hover { transform: translateY(-2px); box-shadow: 0 10px 20px -10px var(--primary); }

        button:disabled { opacity: 0.7; cursor: not-allowed; transform: none; }

        #resultContainer {
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: none;
            animation: slideUp 0.4s ease-out;
        }

        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

        .result-badge { display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; margin-bottom: 1rem; }

        .badge-real { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-fake { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

        .meter-container {
            background: rgba(255, 255, 255, 0.05);
            height: 10px;
            border-radius: 5px;
            margin: 1.5rem 0 0.5rem 0;
            overflow: hidden;
            position: relative;
        }

        .meter-fill { height: 100%; border-radius: 5px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }

        .score-info { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-muted); }

        .explanation {
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 8px;
            margin-top: 1.5rem;
            text-align: left;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #cbd5e1;
            border-left: 3px solid var(--primary);
        }

        .loading-spinner {
            display: none;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Voice Truth</h1>
            <p class="subtitle">Advanced AI Deepfake Detection</p>

            <div class="controls">
                <label>Language</label>
                <select id="languageSelect">
                    <option value="en">English (Universal)</option>
                    <option value="hi">Hindi</option>
                    <option value="ta">Tamil</option>
                    <option value="te">Telugu</option>
                    <option value="ml">Malayalam</option>
                </select>
            </div>

            <div class="upload-area" id="dropArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div id="fileName">Click or Drag Audio File Here</div>
                <input type="file" id="fileInput" accept="audio/*" style="display: none" onchange="handleFileSelect(this)">
            </div>

            <button class="cta" onclick="analyzeAudio()" id="btn">
                <span id="btnText">Analyze Voice</span>
                <div class="loading-spinner" id="spinner"></div>
            </button>

            <div id="resultContainer">
                <div id="resultBadge" class="result-badge"></div>
                <h2 id="resultTitle"></h2>
                
                <div class="meter-container">
                    <div id="meterFill" class="meter-fill"></div>
                </div>
                <div class="score-info">
                    <span>Confidence Score</span>
                    <span id="scoreValue" style="color: white; font-weight: 600">0.00</span>
                </div>

                <div class="explanation" id="explanationText"></div>
            </div>
        </div>
    </div>

    <script>
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                document.getElementById('fileName').innerText = file.name;
                document.getElementById('fileName').style.color = 'var(--text)';
            }
        }

        const dropArea = document.getElementById('dropArea');
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

        ['dragenter', 'dragover'].forEach(name => {
            dropArea.addEventListener(name, () => dropArea.style.borderColor = 'var(--primary)', false);
        });
        
        ['dragleave', 'drop'].forEach(name => {
            dropArea.addEventListener(name, () => dropArea.style.borderColor = 'rgba(255, 255, 255, 0.1)', false);
        });

        dropArea.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            document.getElementById('fileInput').files = files;
            handleFileSelect(document.getElementById('fileInput'));
        }

        async function analyzeAudio() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                const area = document.getElementById('dropArea');
                area.style.borderColor = 'var(--danger)';
                setTimeout(() => area.style.borderColor = 'rgba(255,255,255,0.1)', 1000);
                return;
            }

            const btn = document.getElementById('btn');
            const btnText = document.getElementById('btnText');
            const spinner = document.getElementById('spinner');
            const resultContainer = document.getElementById('resultContainer');
            
            btn.disabled = true;
            btnText.style.display = 'none';
            spinner.style.display = 'block';
            resultContainer.style.display = 'none';

            const formData = new FormData();
            formData.append("file", file);

            try {
                const response = await fetch('/detect/audio-file', {
                    method: 'POST',
                    headers: { 'X-API-Key': 'hackathon-secret-key' },
                    body: formData
                });

                const data = await response.json();
                
                resultContainer.style.display = 'block';
                const isHuman = data.classification === 'HUMAN';
                
                const badge = document.getElementById('resultBadge');
                badge.className = 'result-badge ' + (isHuman ? 'badge-real' : 'badge-fake');
                badge.innerText = isHuman ? 'REAL VOICE' : 'AI GENERATED';

                const title = document.getElementById('resultTitle');
                title.innerText = isHuman ? 'Human Verified' : 'Deepfake Detected';

                const scoreLayout = document.getElementById('scoreValue');
                scoreLayout.innerText = data.confidence.toFixed(4);

                const meter = document.getElementById('meterFill');
                meter.style.width = (data.confidence * 100) + '%';
                meter.style.backgroundColor = isHuman ? 'var(--success)' : 'var(--danger)';

                document.getElementById('explanationText').innerText = data.explanation;

            } catch (error) {
                alert("An error occurred during analysis.");
                console.error(error);
            } finally {
                btn.disabled = false;
                btnText.style.display = 'block';
                spinner.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""

# Endpoints
@app.get("/")
def read_root():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

@app.get("/health")
def health_check():
    return {"status": "active", "model": "Wav2Vec2"}

@app.post("/detect", dependencies=[Depends(get_api_key)])
async def detect_voice(request: VoiceRequest):
    if not request.audio_base64:
        raise HTTPException(status_code=400, detail="Missing audio_base64 field")
    
    result = detector.predict(request.audio_base64)
    
    if result["classification"] == "ERROR":
        raise HTTPException(status_code=500, detail=result["explanation"])
        
    return result

@app.post("/detect/audio-file", dependencies=[Depends(get_api_key)])
async def detect_voice_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
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
