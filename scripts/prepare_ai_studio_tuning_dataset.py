"""
Google AI Studio Tuning Dataset Preparation Script
Generates a JSONL dataset from our 3,277 verified Indian traffic signs
formatted for Google AI Studio Multimodal Model Tuning / Prompt Evaluation.
"""

import os
import json
import base64
from pathlib import Path

def prepare_tuning_jsonl(output_file: str = "artifacts/ai_studio_tuning_dataset.jsonl", max_samples_per_class: int = 5):
    base_dir = Path(__file__).resolve().parent.parent
    train_dir = base_dir / "data" / "train"
    meta_path = base_dir / "data" / "metadata" / "class_names.json"
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
        class_map = {c['key']: c for c in classes}
        
    records = []
    classes_found = list(train_dir.glob("*"))
    print(f"[*] Scanning {len(classes_found)} class directories for tuning samples...")
    
    for cls_dir in classes_found:
        if not cls_dir.is_dir():
            continue
        cls_key = cls_dir.name
        meta = class_map.get(cls_key, {
            "display_name": cls_key.replace('_', ' ').title(),
            "category": "Traffic Sign"
        })
        
        images = list(cls_dir.glob("*.jpg"))[:max_samples_per_class]
        for img_path in images:
            with open(img_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
                
            entry = {
                "system_instruction": "You are an official Indian traffic sign recognition assistant compliant with IRC:67 standards.",
                "input": {
                    "text": "Identify this Indian road traffic sign and provide its regulatory category and legal rule under Indian Motor Vehicles Act.",
                    "image_b64": b64_data[:64] + "...[truncated for size]...",
                    "image_rel_path": str(img_path.relative_to(base_dir))
                },
                "output": {
                    "sign_name": meta["display_name"],
                    "class_key": cls_key,
                    "category": meta["category"],
                    "compliance_standard": "Indian Road Congress (IRC:67)",
                    "rule_description": f"Mandatory compliance sign for {meta['display_name']} on Indian roads."
                }
            }
            records.append(entry)
            
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")
            
    print(f"[+] Prepared {len(records)} tuning samples in: {output_file}")
    return len(records)

if __name__ == "__main__":
    prepare_tuning_jsonl()
