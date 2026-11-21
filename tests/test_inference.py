import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.inference import CLASS_LABELS, load_uploaded_image, predict_images
from src.model import build_densenet121, load_densenet121


class InferenceSmokeTests(unittest.TestCase):
    def test_class_order(self):
        self.assertEqual(
            CLASS_LABELS,
            ["B12", "Iron", "Vitamin_A", "Vitamin_C", "Vitamin_D", "Zinc"],
        )

    def test_uploaded_image_conversion(self):
        image = Image.new("RGBA", (80, 40), (120, 30, 90, 200))
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        parsed = load_uploaded_image(buffer.getvalue())
        self.assertEqual(parsed.mode, "RGB")
        self.assertEqual(parsed.size, (80, 40))

    def test_model_checkpoint_and_prediction_shape(self):
        # This is only a smoke test of the inference path using random weights.
        # It does NOT represent the project's trained model or meaningful predictions.
        torch.set_num_threads(2)
        model = build_densenet121()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "smoke_test_only.pth"
            torch.save(model.state_dict(), checkpoint)
            loaded = load_densenet121(checkpoint, device=torch.device("cpu"))
            sample = Image.new("RGB", (224, 224), (120, 80, 140))
            outputs = predict_images(loaded, [sample], device=torch.device("cpu"), batch_size=1)
            self.assertEqual(len(outputs), 1)
            self.assertIn(outputs[0]["predicted_label"], CLASS_LABELS)
            self.assertAlmostEqual(sum(outputs[0]["scores"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
