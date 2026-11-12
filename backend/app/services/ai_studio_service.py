"""
Google AI Studio Integration Service
Connects to Google AI Studio Gemini API for multimodal Indian traffic sign
analysis, regulatory interpretation, and driver assistance advice.
"""

import os
import json
import base64
from typing import Dict, Any, Optional
from pathlib import Path

class AIStudioService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("AI_STUDIO_MODEL", "gemini-1.5-flash")
        
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def analyze_sign_multimodal(self, image_bytes: bytes, local_prediction: Optional[str] = None) -> Dict[str, Any]:
        """
        Uses Google AI Studio Gemini model to perform comprehensive vision analysis
        and provide Indian Motor Vehicles Act regulatory interpretation.
        """
        if not self.api_key:
            return {
                "configured": False,
                "message": "Google AI Studio API key not found. Get a free API key at https://aistudio.google.com/ and set GEMINI_API_KEY.",
                "guide_url": "https://aistudio.google.com/"
            }
            
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            
            prompt = (
                "You are an expert Indian Traffic Police Assistant and Computer Vision specialist. "
                "Analyze this road image captured on an Indian road.\n"
                f"Local BanTrafficNet model predicted: '{local_prediction or 'Unknown'}'.\n\n"
                "Provide your response in strictly valid JSON format with these exact keys:\n"
                "{\n"
                '  "sign_detected": "Name of the Indian traffic sign",\n'
                '  "category": "Mandatory / Regulatory OR Cautionary / Warning OR Informatory",\n'
                '  "confidence": "High / Medium / Low",\n'
                '  "irc_rule": "Specific rule according to IRC:67 / Motor Vehicles Act 1988",\n'
                '  "driver_action": "Clear, actionable advice for the driver on Indian roads",\n'
                '  "penalty_risk": "Applicable fine or legal penalty if violated"\n'
                "}"
            )
            
            contents = [
                {"mime_type": "image/jpeg", "data": image_bytes},
                prompt
            ]
            
            response = model.generate_content(contents)
            text = response.text.strip()
            
            # Clean markdown code block formatting if returned
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            data = json.loads(text.strip())
            return {
                "configured": True,
                "ai_studio_model": self.model_name,
                "analysis": data
            }
        except Exception as e:
            return {
                "configured": True,
                "error": str(e),
                "message": "Failed to call Google AI Studio Gemini API."
            }

ai_studio_service = AIStudioService()
