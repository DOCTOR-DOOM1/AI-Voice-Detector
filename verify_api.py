import requests
import base64
import json

# Configuration
API_URL = "http://127.0.0.1:8001/detect"
API_KEY = "hackathon-secret-key"

def create_dummy_mp3():
    """Create a dummy header-only MP3/Audio file for testing logic if no file exists"""
    # This is not a real MP3, but enough bytes for the logic to try reading
    # In real usage, you'd read an actual file: with open("test.mp3", "rb") as f: ...
    return b"ID3" + b"\x00"*20 + b"\xFF\xFB"*100 # minimal fake structure

def test_api():
    print("Generating dummy audio...")
    audio_bytes = create_dummy_mp3()
    b64_string = base64.b64encode(audio_bytes).decode('utf-8')
    
    # We expect librosa to fail on this garbage data, but we want to confirm the API *receives* and *handles* it.
    # The API should return 500 or 200 depending on if librosa can interpret it as silence or errors out.
    # Ideally, we should provide a tiny valid Base64 string if possible, or just expect the error handling to work.
    
    payload = {"audio_base64": b64_string}
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {API_URL}...")
    try:
        # Note: This requires the server to be running. 
        # Since I cannot run background server + script in one go easily without blocking,
        # This script is for the USER to run.
        response = requests.post(API_URL, json=payload, headers=headers)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print(f"Request failed (Server likely not running): {e}")

if __name__ == "__main__":
    test_api()
