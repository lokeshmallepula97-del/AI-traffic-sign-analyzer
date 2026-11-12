"""
End-to-End System Test Suite for Indian Traffic Sign Recognition using BanTrafficNet.
Validates:
1. BanTrafficNet parameter formula (4.6M params)
2. Verified Indian dataset threshold (>= 1,001 original images)
3. FastAPI/Flask endpoints: /health, /model/status, /dataset/summary, /dataset/classes, /dataset/sample
4. Live inference via /predict with real roadside test signs
5. Edge case handling (invalid files, empty requests)
"""

import io
import json
import unittest
from pathlib import Path

from backend.app.main import app
from backend.app.models.bantrafficnet import calculate_parameter_count

class TestBanTrafficNetSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.base_dir = Path(__file__).resolve().parent.parent

    def test_01_parameter_formula_verification(self):
        """Verify BanTrafficNet parameter scaling matching reference paper."""
        # 48 classes (paper)
        p48 = calculate_parameter_count(48)
        self.assertEqual(p48["total_parameters"], 4641488, "Reference paper 48-class parameter mismatch!")
        
        # 56 classes (Indian dataset)
        p56 = calculate_parameter_count(56)
        expected_56 = 4629152 + (257 * 56)
        self.assertEqual(p56["total_parameters"], expected_56)
        self.assertEqual(p56["total_parameters"], 4643544)

    def test_02_dataset_verification_threshold(self):
        """Verify official Indian dataset meets the mandatory >= 1,001 original images gate."""
        summary_file = self.base_dir / "data" / "metadata" / "dataset_summary.json"
        self.assertTrue(summary_file.exists(), "dataset_summary.json not found!")
        
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.assertGreaterEqual(data["total_original_images"], 1001, "Gate Failed: Fewer than 1,001 original images!")
        self.assertEqual(data["total_original_images"], 3277)
        self.assertEqual(data["total_classes"], 56)
        self.assertTrue(data["minimum_requirement_met"])

    def test_03_health_endpoint(self):
        """Test GET /api/health returns online status."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["architecture"], "BanTrafficNet")

    def test_04_model_status_endpoint(self):
        """Test GET /api/model/status reports active BanTrafficNet engine."""
        res = self.client.get("/api/model/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["total_classes"], 56)
        self.assertEqual(data["total_parameters"], 4643544)
        self.assertEqual(data["input_resolution"], "96x96x3 (RGB)")

    def test_05_dataset_endpoints(self):
        """Test GET /api/dataset/summary and /api/dataset/classes."""
        res_sum = self.client.get("/api/dataset/summary")
        self.assertEqual(res_sum.status_code, 200)
        sum_data = res_sum.get_json()
        self.assertEqual(sum_data["total_classes"], 56)
        
        res_cls = self.client.get("/api/dataset/classes")
        self.assertEqual(res_cls.status_code, 200)
        classes = res_cls.get_json()
        self.assertEqual(len(classes), 56)
        
        # Test sample image fetch
        first_key = classes[0]["key"]
        res_sample = self.client.get(f"/api/dataset/sample/{first_key}")
        self.assertEqual(res_sample.status_code, 200)
        self.assertEqual(res_sample.content_type, "image/jpeg")

    def test_06_live_prediction_endpoint(self):
        """Test POST /api/predict with genuine road image."""
        test_images = list((self.base_dir / "data" / "test").rglob("*.jpg"))
        self.assertGreater(len(test_images), 0, "No test images found in data/test!")
        
        sample_path = test_images[0]
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
            
        data = {"file": (io.BytesIO(img_bytes), sample_path.name)}
        res = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        pred = res.get_json()
        self.assertTrue(pred["success"])
        self.assertIn("predicted_class", pred)
        self.assertIn("confidence", pred)
        self.assertIn("category", pred)
        self.assertEqual(len(pred["top_predictions"]), 3)

    def test_07_invalid_input_handling(self):
        """Test POST /api/predict rejects empty or corrupt uploads."""
        # Missing file field
        res_empty = self.client.post("/api/predict", data={})
        self.assertEqual(res_empty.status_code, 400)
        
        # Corrupt bytes
        corrupt_data = {"file": (io.BytesIO(b"not_an_image_binary"), "test.jpg")}
        res_corrupt = self.client.post("/api/predict", data=corrupt_data, content_type="multipart/form-data")
        self.assertEqual(res_corrupt.status_code, 500)
        self.assertFalse(res_corrupt.get_json()["success"])

if __name__ == "__main__":
    unittest.main()
