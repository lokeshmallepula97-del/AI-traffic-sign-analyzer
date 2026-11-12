# Google AI Studio Deployment & Integration Guide
### Indian Traffic Sign Recognition System with BanTrafficNet + Gemini Multimodal Copilot

---

## 1. Overview: How Google AI Studio Powers This Project

**Google AI Studio** ([aistudio.google.com](https://aistudio.google.com/)) is Google's web-based platform for rapid prototyping, multimodal model fine-tuning, and production API key generation with state-of-the-art Gemini models (Gemini 1.5 Flash, Gemini 1.5 Pro, and Gemini 2.0).

In this project, we implement a **Hybrid Dual-Engine Architecture**:
1. **BanTrafficNet Edge Engine:** Ultra-low latency ($<10$ ms), lightweight (4.6M params) CNN running locally for instant classification of the 56 Indian sign categories.
2. **Google AI Studio Cloud Engine:** Gemini 1.5 Flash multimodal intelligence providing deep contextual reasoning, Indian Motor Vehicles Act legal penalty citations, and real-time driver coaching based on road conditions.

---

## 2. Option 1: Live Deployment with Google AI Studio API Key

This allows immediate, zero-cost deployment of the Gemini multimodal intelligence layer.

### Step 1: Obtain a Free Google AI Studio API Key
1. Visit **[https://aistudio.google.com/](https://aistudio.google.com/)** and sign in with your Google account.
2. In the top-left menu or header, click **"Get API key"**.
3. Click **"Create API key in new project"** (or select an existing Google Cloud project).
4. Copy your generated API key (it begins with `AIzaSy...`).

### Step 2: Configure Environment Variables
Open or create `.env` in the project root:
```env
PORT=8000
HOST=127.0.0.1
ENVIRONMENT=production
GEMINI_API_KEY=AIzaSyYourGeneratedGoogleAIStudioKeyHere
AI_STUDIO_MODEL=gemini-1.5-flash
```

### Step 3: Run the Application
```powershell
python backend/app/main.py
```
Open `http://localhost:8000`. The application will automatically route traffic sign images to both BanTrafficNet and Google AI Studio!

---

## 3. Option 2: Fine-Tuning a Custom Gemini Model in Google AI Studio

You can fine-tune a specialized Indian Traffic Sign variant of Gemini directly inside Google AI Studio:

1. Navigate to **[aistudio.google.com/app/tuned_models](https://aistudio.google.com/app/tuned_models)**.
2. Click **"New tuned model"**.
3. **Select Base Model:** Choose `gemini-1.5-flash` (fast, cost-effective).
4. **Import Training Data:** Upload the pre-generated dataset file created in our project:
   ```
   artifacts/ai_studio_tuning_dataset.jsonl
   ```
5. **Set Hyperparameters:**
   * Epochs: 4–10
   * Batch size: 4
   * Learning rate multiplier: 1.0
6. Click **"Tune"**. Google's cloud infrastructure will fine-tune the model on your Indian dataset for free.
7. Once finished, copy the tuned model resource identifier (e.g., `tunedModels/indian-traffic-signs-abc123`).
8. Update `.env`:
   ```env
   AI_STUDIO_MODEL=tunedModels/indian-traffic-signs-abc123
   ```

---

## 4. Option 3: Full Cloud Deployment via Google Cloud Run

To host the entire full-stack web app on Google's infrastructure:

### Step 1: Create a Containerfile / Dockerfile
Save as `Dockerfile` in the project root:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt google-generativeai

COPY . .

EXPOSE 8000
ENV PORT=8000
CMD ["python", "backend/app/main.py"]
```

### Step 2: Deploy to Google Cloud Run
Using the Google Cloud CLI (`gcloud`):
```bash
gcloud run deploy indian-traffic-sign-ai \
    --source . \
    --platform managed \
    --region asia-south1 \
    --allow-unauthenticated \
    --set-env-vars GEMINI_API_KEY="your-google-ai-studio-key"
```
Within 2 minutes, Google Cloud Run will issue a permanent public HTTPS URL (e.g., `https://indian-traffic-sign-ai-xyz-as.a.run.app`).

---

## 5. Google AI Studio Prompt Template

If you want to test traffic signs directly in the Google AI Studio web interface:

### System Instructions:
```text
You are an expert Indian Traffic Police Assistant and Autonomous Driving Vision Specialist.
You strictly classify traffic signs according to the Indian Road Congress (IRC:67) standards
and provide legal guidance under the Indian Motor Vehicles Act, 1988 (and 2019 Amendment).

Respond in clean JSON:
{
  "sign_detected": "Official Name of Sign",
  "category": "Mandatory / Regulatory | Cautionary / Warning | Informatory",
  "confidence": "High / Medium / Low",
  "irc_rule": "Precise IRC:67 clause or MV Act regulation",
  "driver_action": "Actionable instructions for the driver",
  "penalty_risk": "Applicable fine (e.g., ₹500 - ₹2000 under Section 177/184)"
}
```

### User Input:
Attach any roadside image from `data/test/` and run the model in AI Studio!
