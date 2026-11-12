"""
Inference Service for BanTrafficNet.
Handles image ingestion, preprocessing (96x96 RGB, [0, 1] normalization),
high-speed forward pass via cv2.dnn, top-3 confidence ranking,
and IRC:67 regulatory metadata enrichment.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

class InferenceService:
    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        # file is at backend/app/services/inference_service.py -> 3 parents up is backend/, 4 parents up is project root
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.model_path = Path(model_path) if model_path else (self.base_dir / "artifacts" / "model" / "bantrafficnet_indian.onnx")
        self.metadata_path = Path(metadata_path) if metadata_path else (self.base_dir / "data" / "metadata" / "class_names.json")
        
        self.net = None
        self.classes = []
        self.class_lookup = {}
        self.is_loaded = False
        self.load_metadata()
        self.load_model()
        
    def load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)
                self.class_lookup = {item['id']: item for item in self.classes}
                self.key_lookup = {item['key']: item for item in self.classes}
            print(f"[+] InferenceService loaded {len(self.classes)} class definitions.")
        else:
            print(f"[!] Metadata file not found at {self.metadata_path}")
            
    def load_model(self) -> bool:
        if self.model_path.exists():
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.is_loaded = True
                print(f"[+] Loaded BanTrafficNet ONNX model from: {self.model_path}")
                return True
            except Exception as e:
                print(f"[!] Error loading ONNX model: {e}")
                self.is_loaded = False
                return False
        else:
            print(f"[*] Model file not yet present at: {self.model_path}")
            self.is_loaded = False
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.is_loaded,
            "architecture": "BanTrafficNet (Scientific Reports 2026)",
            "model_format": "ONNX (OpenCV DNN Engine)",
            "model_path": str(self.model_path),
            "input_resolution": "96x96x3 (RGB)",
            "total_classes": len(self.classes),
            "total_parameters": 4643544,
            "runtime_environment": "Python 3 + OpenCV DNN Engine"
        }

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        # 1. Decode bytes into image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image. Unsupported or corrupted image format.")
            
        # 2. BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Resize to 96x96 (exact BanTrafficNet specification)
        img_resized = cv2.resize(img_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        
        # 4. Rescale pixel values to [0, 1] as defined by Eq. (4): X / 255.0
        img_norm = img_resized.astype(np.float32) / 255.0
        
        # 5. Convert HWC to NCHW: (1, 3, 96, 96)
        blob = np.transpose(img_norm, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                "success": False,
                "error": "BanTrafficNet model is not loaded. Please upload model weights or run training.",
                "model_status": "Training pending"
            }
            
        try:
            t0 = time.perf_counter()
            blob = self.preprocess_image(image_bytes)
            
            # Forward pass
            self.net.setInput(blob)
            preds = self.net.forward()[0] # Shape (56,)
            inference_ms = round((time.perf_counter() - t0) * 1000, 2)
            
            # Sort top 3
            top_indices = np.argsort(preds)[::-1][:3]
            
            top_predictions = []
            for rank, idx in enumerate(top_indices):
                meta = self.class_lookup.get(int(idx), {
                    "id": int(idx),
                    "key": f"class_{idx}",
                    "display_name": f"Class {idx}",
                    "category": "Unknown"
                })
                conf = float(preds[idx])
                top_predictions.append({
                    "rank": rank + 1,
                    "class_id": int(idx),
                    "class_name": meta["display_name"],
                    "class_key": meta["key"],
                    "category": meta["category"],
                    "confidence": round(conf * 100, 2),
                    "probability": round(conf, 4)
                })
                
            primary = top_predictions[0]
            
            # IRC rule explanation
            irc_rules = {
                "Mandatory / Regulatory": "Drivers are legally bound by Indian Motor Vehicles Act to adhere strictly to this sign. Non-compliance invites penal action.",
                "Cautionary / Warning": "Alerts driver to upcoming physical hazards or geometric changes on Indian roads. Reduce speed and proceed with caution.",
                "Informatory": "Provides guidance on roadside services, facility directions, and geographic navigation per IRC:67 guidelines."
            }
            
            return {
                "success": True,
                "predicted_class": primary["class_name"],
                "class_key": primary["class_key"],
                "confidence": primary["confidence"],
                "category": primary["category"],
                "regulatory_meaning": irc_rules.get(primary["category"], "Indian Road Congress (IRC:67) compliant signage."),
                "top_predictions": top_predictions,
                "inference_time_ms": inference_ms,
                "input_dimensions": "96x96x3",
                "model": "BanTrafficNet"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Global singleton
inference_service = InferenceService()
