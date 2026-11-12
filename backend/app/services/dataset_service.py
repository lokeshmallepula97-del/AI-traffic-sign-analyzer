"""
Dataset Service for Indian Traffic Sign Recognition System.
Provides summary statistics, class catalogs, sample previews,
and integrity verification data.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

class DatasetService:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.meta_dir = self.base_dir / "data" / "metadata"
        self.original_dir = self.base_dir / "data" / "original"
        self.train_dir = self.base_dir / "data" / "train"
        self.val_dir = self.base_dir / "data" / "validation"
        self.test_dir = self.base_dir / "data" / "test"
        
    def get_summary(self) -> Dict[str, Any]:
        summary_path = self.meta_dir / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "error": "Dataset summary not yet generated.",
            "total_original_images": 0,
            "total_classes": 0
        }
        
    def get_classes(self) -> List[Dict[str, Any]]:
        classes_path = self.meta_dir / "class_names.json"
        if classes_path.exists():
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = json.load(f)
                # Enrich with sample image URL / existence
                for c in classes:
                    cls_key = c['key']
                    samples = list((self.original_dir / cls_key).glob("*.jpg"))
                    c["available_samples_count"] = len(samples)
                    c["sample_image_url"] = f"/api/dataset/sample/{cls_key}" if samples else None
                return classes
        return []

    def get_sample_image_path(self, class_key: str) -> Path | None:
        cls_dir = self.original_dir / class_key
        if cls_dir.exists():
            samples = list(cls_dir.glob("*.jpg"))
            if samples:
                return samples[0]
        return None

dataset_service = DatasetService()
