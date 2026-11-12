"""
Organizes raw downloaded Indian traffic sign images into class-specific folders,
performs verified stratified splitting, and exports canonical metadata.
"""

import os
import re
import json
import shutil
import hashlib
from pathlib import Path
from collections import Counter
from PIL import Image

def parse_class_name(filename_stem: str) -> str:
    m = re.match(r'^(.*?)(?:-\d+.*|\.rf\..*)', filename_stem)
    if m:
        return m.group(1).replace('-', '_').strip('_')
    return filename_stem

def get_irc_category(cls_name: str) -> str:
    regulatory = {
        'stop', 'give_way', 'no_entry', 'straight_prohibited', 'no_parking',
        'no_stopping_or_standing', 'left_turn_prohibited', 'right_turn_prohibited',
        'u_turn_prohibited', 'overtaking_prohibited', 'horn_prohibited',
        'all_motor_vehicle_prohibited', 'speed_limit_30', 'speed_limit_50',
        'speed_limit_100', 'axle_load_limit', 'height_limit', 'width_limit',
        'compulsary_ahead', 'compulsary_keep_left', 'compulsary_keep_right',
        'compulsary_turn_left_ahead', 'compulsary_turn_right_ahead',
        'pass_either_side', 'restriction_ends'
    }
    informatory = {'hospital_ahead', 'petrol_pump_ahead', 'u_turn'}
    
    if cls_name in regulatory:
        return "Mandatory / Regulatory"
    elif cls_name in informatory:
        return "Informatory"
    else:
        return "Cautionary / Warning"

def format_display_name(cls_name: str) -> str:
    words = cls_name.split('_')
    formatted = []
    for w in words:
        if w.lower() == 'compulsary':
            formatted.append('Compulsory')
        elif w.isdigit():
            formatted.append(w)
        else:
            formatted.append(w.capitalize())
    return ' '.join(formatted)

def hash_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def organize():
    raw_dir = Path("data/raw_extracted")
    original_dir = Path("data/original")
    train_dir = Path("data/train")
    val_dir = Path("data/validation")
    test_dir = Path("data/test")
    meta_dir = Path("data/metadata")
    
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather all unique original images
    all_raw_files = list(raw_dir.rglob("*.jpg"))
    print(f"[*] Found {len(all_raw_files)} total image files across raw splits")
    
    seen_hashes = {}
    valid_images = []
    
    for f in all_raw_files:
        try:
            with Image.open(f) as img:
                img.verify()
            h = hash_file(f)
            if h not in seen_hashes:
                seen_hashes[h] = f
                cls = parse_class_name(f.stem)
                valid_images.append((f, cls, h))
        except Exception as e:
            print(f"[!] Warning: corrupted image {f}: {e}")
            
    print(f"[+] Total verified unique original images: {len(valid_images)}")
    assert len(valid_images) >= 1001, f"Failed requirement: {len(valid_images)} < 1001"
    
    # 2. Group by class
    class_groups = {}
    for src_file, cls, h in valid_images:
        class_groups.setdefault(cls, []).append((src_file, h))
        
    sorted_classes = sorted(class_groups.keys())
    print(f"[+] Total verified classes: {len(sorted_classes)}")
    
    # 3. Create class metadata
    class_metadata = []
    for idx, cls in enumerate(sorted_classes):
        cat = get_irc_category(cls)
        display_name = format_display_name(cls)
        count = len(class_groups[cls])
        class_metadata.append({
            "id": idx,
            "key": cls,
            "display_name": display_name,
            "category": cat,
            "original_count": count
        })
        
    with open(meta_dir / "class_names.json", "w") as f:
        json.dump(class_metadata, f, indent=2)
    print(f"[+] Saved {meta_dir / 'class_names.json'}")
    
    # 4. Copy to data/original/<class_name>/
    print("[*] Copying images to data/original/<class_name>/...")
    for cls in sorted_classes:
        cls_dir = original_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i, (src_file, h) in enumerate(class_groups[cls]):
            dest_file = cls_dir / f"{cls}_{i+1:04d}.jpg"
            shutil.copy2(src_file, dest_file)
            
    # 5. Perform stratified split (75% Train, 12.5% Val, 12.5% Test)
    print("[*] Creating stratified train (75%), validation (12.5%), and test (12.5%) partitions...")
    train_count = 0
    val_count = 0
    test_count = 0
    
    for cls in sorted_classes:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)
        
        items = list((original_dir / cls).glob("*.jpg"))
        n = len(items)
        n_val = max(1, int(round(n * 0.125)))
        n_test = max(1, int(round(n * 0.125)))
        n_train = n - n_val - n_test
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]
        
        for item in train_items:
            shutil.copy2(item, train_dir / cls / item.name)
            train_count += 1
        for item in val_items:
            shutil.copy2(item, val_dir / cls / item.name)
            val_count += 1
        for item in test_items:
            shutil.copy2(item, test_dir / cls / item.name)
            test_count += 1
            
    print(f"[+] Dataset Split Summary:")
    print(f"    Train:      {train_count} images ({train_count/len(valid_images)*100:.1f}%)")
    print(f"    Validation: {val_count} images ({val_count/len(valid_images)*100:.1f}%)")
    print(f"    Test:       {test_count} images ({test_count/len(valid_images)*100:.1f}%)")
    print(f"    Total:      {train_count + val_count + test_count} images")
    
    summary = {
        "dataset_name": "Indian Traffic Signboards Dataset (IRC:67 Compliant)",
        "source": "Roboflow Universe - Major Project vu7ji (Authentic Indian Road Captures)",
        "license": "MIT License",
        "total_original_images": len(valid_images),
        "total_classes": len(sorted_classes),
        "split": {
            "train": train_count,
            "validation": val_count,
            "test": test_count
        },
        "minimum_requirement_met": len(valid_images) >= 1001,
        "input_target_resolution": [96, 96, 3],
        "categories_breakdown": {
            "Mandatory / Regulatory": sum(1 for c in class_metadata if c["category"] == "Mandatory / Regulatory"),
            "Cautionary / Warning": sum(1 for c in class_metadata if c["category"] == "Cautionary / Warning"),
            "Informatory": sum(1 for c in class_metadata if c["category"] == "Informatory")
        }
    }
    
    with open(meta_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved {meta_dir / 'dataset_summary.json'}")
    
    return summary, class_metadata

if __name__ == "__main__":
    organize()
