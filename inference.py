import torch
import librosa
import numpy as np
import logging
import io
import base64
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 16000
MODEL_NAME = "mo-thecreator/Deepfake-audio-detection"

class VoiceDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model: {MODEL_NAME} (Device: {self.device})")
        logger.info("This may take a while on first run deeply depending on internet speed...")
        
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(self.device)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _decode_audio(self, base64_string):
        """Decode base64 string to audio time series"""
        try:
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            
            audio_bytes = base64.b64decode(base64_string)
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
            return y
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            raise ValueError("Invalid audio data")

    def predict(self, base64_audio):
        try:
            # 1. Decode
            y = self._decode_audio(base64_audio)
            
            # 2. Preprocess (Transformers Feature Extractor)
            # The model expects input values, not raw LFCC/MFCC tensors we made manually before
            inputs = self.feature_extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # 3. Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                
                # Check config labels. Usually 0=Fake, 1=Real or vice versa.
                # For `mo-thecreator/Deepfake-audio-detection`: 
                # Label 0: "real", Label 1: "fake" (Need to confirm via id2label usually, but let's assume standard behavior or inspect)
                # Correction: Most Deepfake models: 1 is Fake (positive class). 
                # Let's inspect self.model.config.id2label dynamically if possible, but hardcoding for now based on typical behavior.
                # If the model has config with labels:
                id2label = self.model.config.id2label
                
                # Get the highest probability class
                predicted_id = torch.argmax(logits, dim=-1).item()
                predicted_label = id2label[predicted_id]
                score = probs[0][predicted_id].item()
                
                # Normalize output to our API standard
                if "fake" in predicted_label.lower() or "spoof" in predicted_label.lower():
                     classification = "AI_GENERATED"
                     fake_score = score
                else:
                     classification = "HUMAN"
                     fake_score = 1.0 - score # just for reference if needed
                
            return {
                "classification": classification,
                "confidence": round(score, 4),
                "explanation": f"Classified by AI Model as '{predicted_label}'."
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "classification": "ERROR",
                "confidence": 0.0,
                "explanation": str(e)
            }

# Singleton instance
# NOTE: This will trigger download on import!
try:
    detector = VoiceDetector()
except Exception:
    # If network fails, we shouldn't crash independent imports, but app startup will fail.
    detector = None

