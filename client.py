import requests
import argparse
import sys
import os

# Configuration
API_URL = "http://127.0.0.1:8001/detect/audio-file"
API_KEY = "hackathon-secret-key"

def check_voice(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Uploading '{file_path}' to {API_URL}...")
    
    headers = {"X-API-Key": API_KEY}
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(API_URL, headers=headers, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*40)
            print(f"ANALYSIS RESULT: {result['classification']}")
            print("="*40)
            print(f"Confidence:  {result['confidence'] * 100:.2f}%")
            print(f"Explanation: {result['explanation']}")
            print("="*40 + "\n")
        else:
            print(f"Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is 'python app.py' running?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_audio_file>")
        print("Example: python client.py test_audio.mp3")
    else:
        check_voice(sys.argv[1])
