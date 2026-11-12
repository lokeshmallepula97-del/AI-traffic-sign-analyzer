export interface PredictionItem {
  rank: number;
  class_id: number;
  class_name: string;
  class_key: string;
  category: string;
  confidence: number;
  probability: number;
}

export interface PredictionResponse {
  success: boolean;
  predicted_class?: string;
  class_key?: string;
  confidence?: number;
  category?: string;
  regulatory_meaning?: string;
  top_predictions?: PredictionItem[];
  inference_time_ms?: number;
  input_dimensions?: string;
  model?: string;
  error?: string;
  model_status?: string;
}

export interface DatasetClass {
  id: number;
  key: string;
  display_name: string;
  category: string;
  original_count: number;
  available_samples_count?: number;
  sample_image_url?: string;
}

export interface DatasetSummary {
  dataset_name: string;
  source: string;
  license: string;
  total_original_images: number;
  total_classes: number;
  split: {
    train: number;
    validation: number;
    test: number;
  };
  minimum_requirement_met: boolean;
  input_target_resolution: number[];
  categories_breakdown: {
    [key: string]: number;
  };
}

export interface ModelStatus {
  model_loaded: boolean;
  architecture: string;
  model_format: string;
  model_path: string;
  input_resolution: string;
  total_classes: number;
  total_parameters: number;
  runtime_environment: string;
}

export interface EvaluationMetrics {
  status?: string;
  model_architecture?: string;
  dataset_name?: string;
  total_params?: number;
  input_resolution?: string;
  training_epochs?: number;
  best_val_acc?: number;
  test_acc?: number;
  macro_precision?: number;
  macro_recall?: number;
  macro_f1?: number;
  mean_inference_latency_ms?: number;
  history?: {
    train_loss: number[];
    train_acc: number[];
    val_loss: number[];
    val_acc: number[];
  };
  confusion_matrix?: number[][];
  per_class_metrics?: {
    [key: string]: {
      precision: number;
      recall: number;
      f1_score: number;
      test_samples: number;
    };
  };
  message?: string;
}

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/model/status`);
  return res.json();
}

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  const res = await fetch(`${API_BASE}/dataset/summary`);
  return res.json();
}

export async function fetchDatasetClasses(): Promise<DatasetClass[]> {
  const res = await fetch(`${API_BASE}/dataset/classes`);
  return res.json();
}

export async function fetchMetrics(): Promise<EvaluationMetrics> {
  const res = await fetch(`${API_BASE}/metrics`);
  return res.json();
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

export async function uploadModelFile(file: File) {
  const formData = new FormData();
  formData.append('model', file);
  const res = await fetch(`${API_BASE}/model/upload`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

export interface AIStudioAnalysisResponse {
  configured: boolean;
  message?: string;
  guide_url?: string;
  ai_studio_model?: string;
  analysis?: {
    sign_detected: string;
    category: string;
    confidence: string;
    irc_rule: string;
    driver_action: string;
    penalty_risk: string;
  };
  error?: string;
}

export async function analyzeWithAIStudio(file: File, localPrediction?: string): Promise<AIStudioAnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (localPrediction) {
    formData.append('local_prediction', localPrediction);
  }
  const res = await fetch(`${API_BASE}/ai-studio/analyze`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}
