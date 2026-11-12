"""
FastAPI / REST Backend Application for Indian Traffic Sign Recognition using BanTrafficNet.
Provides all API endpoints specified in the master project architecture.
"""

import os
import io
import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from backend.app.services.inference_service import inference_service
from backend.app.services.dataset_service import dataset_service
from backend.app.services.ai_studio_service import ai_studio_service

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "artifacts" / "reports"
MODELS_DIR = BASE_DIR / "artifacts" / "model"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Indian Traffic Sign Recognition API",
        "architecture": "BanTrafficNet",
        "framework": "Deep Learning / OpenCV DNN Engine",
        "model_loaded": inference_service.is_loaded
    })

@app.route("/api/model/status", methods=["GET"])
def model_status():
    return jsonify(inference_service.get_status())

@app.route("/api/dataset/summary", methods=["GET"])
def get_dataset_summary():
    return jsonify(dataset_service.get_summary())

@app.route("/api/dataset/classes", methods=["GET"])
def get_dataset_classes():
    return jsonify(dataset_service.get_classes())

@app.route("/api/dataset/sample/<class_key>", methods=["GET"])
def get_dataset_sample(class_key):
    path = dataset_service.get_sample_image_path(class_key)
    if path and path.exists():
        return send_file(path, mimetype="image/jpeg")
    return jsonify({"error": f"Sample not found for class: {class_key}"}), 404

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    metrics_file = REPORTS_DIR / "evaluation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
            
    # If full training run not yet executed, report status transparently
    summary = dataset_service.get_summary()
    return jsonify({
        "status": "pending_or_partial",
        "training_environment": "Google Colab (Recommended GPU accelerator)",
        "model_architecture": "BanTrafficNet",
        "total_parameters": 4643544,
        "classes_count": summary.get("total_classes", 56),
        "total_original_images": summary.get("total_original_images", 3277),
        "split_summary": summary.get("split", {}),
        "message": "Full 100-epoch training notebook ready for Google Colab execution in notebooks/Indian_Traffic_Sign_BanTrafficNet.ipynb."
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in request."}), 400
        
    file = request.files.get("file") or request.files.get("image")
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400
        
    try:
        image_bytes = file.read()
        res = inference_service.predict(image_bytes)
        status_code = 200 if res.get("success") else 500
        return jsonify(res), status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/ai-studio/analyze", methods=["POST"])
def ai_studio_analyze():
    if "file" not in request.files and "image" not in request.files:
        return jsonify({"configured": False, "error": "No image file provided."}), 400
        
    file = request.files.get("file") or request.files.get("image")
    local_pred = request.form.get("local_prediction", None)
    
    try:
        image_bytes = file.read()
        res = ai_studio_service.analyze_sign_multimodal(image_bytes, local_prediction=local_pred)
        return jsonify(res)
    except Exception as e:
        return jsonify({"configured": False, "error": str(e)}), 500

@app.route("/api/model/upload", methods=["POST"])
def upload_model():
    if "model" not in request.files:
        return jsonify({"success": False, "error": "No model file uploaded."}), 400
        
    file = request.files["model"]
    filename = file.filename
    if not filename.endswith(('.onnx', '.keras', '.pth', '.tflite')):
        return jsonify({"success": False, "error": "Supported model extensions: .onnx, .keras, .pth, .tflite"}), 400
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = MODELS_DIR / filename
    file.save(dest_path)
    
    # If ONNX, hot-reload inference engine
    if filename.endswith('.onnx'):
        inference_service.model_path = dest_path
        reloaded = inference_service.load_model()
        return jsonify({
            "success": True,
            "message": f"Model {filename} uploaded and activated successfully.",
            "hot_reloaded": reloaded
        })
        
    return jsonify({
        "success": True,
        "message": f"Model {filename} uploaded. Note: for real-time web inference, upload .onnx format or convert via notebook.",
        "file_path": str(dest_path)
    })

# Serve frontend build in production / single-server mode
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"error": "Endpoint not found"}), 404
        
    frontend_dist = BASE_DIR / "frontend" / "dist"
    target = frontend_dist / path
    if target.exists() and not target.is_dir():
        return send_file(target)
        
    index_file = frontend_dist / "index.html"
    if index_file.exists():
        return send_file(index_file)
        
    return jsonify({
        "status": "Indian Traffic Sign Recognition API is running",
        "notice": "Frontend build not detected in frontend/dist. Start frontend dev server via 'cd frontend; npm run dev'."
    })

if __name__ == "__main__":
    print("[*] Starting Indian Traffic Sign Recognition API on http://localhost:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
