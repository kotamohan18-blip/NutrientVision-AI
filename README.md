# NutrientVision AI

A compact Streamlit application for the project's six-class image-label classification experiment. The app deploys **one model only: the trained DenseNet121 checkpoint**, selected because the project report records the highest held-out test accuracy for this model (90.91%).

> **Research demonstration only.** The dataset's image provenance and laboratory confirmation of labels were not provided. The app predicts one of the supplied folder labels; it does not diagnose vitamin or mineral deficiencies.

## 1. Features

- Single-image upload and preview (JPG/JPEG/PNG).
- Six-class prediction with model score breakdown.
- Batch image inference with CSV export.
- Model information, dataset counts, training setup, and evaluation caveats.
- Responsive dark UI.
- CPU inference by default; CUDA is used automatically when available.
- Docker packaging and health check.

## 2. Required model checkpoint

The notebook creates this file in Google Drive:

`Vitamin_deficiency_Dataset/model_results/densenet121_best.pth`

Copy that exact file into this project's:

`models/densenet121_best.pth`

Example from Google Colab:

```python
from google.colab import files

files.download(
    "/content/drive/MyDrive/Vitamin_deficiency_Dataset/model_results/densenet121_best.pth"
)
```

Then place the downloaded file in the `models/` directory.

**Important:** the checkpoint is not included in this source package because it was not among the uploaded files available for this build. Until you add the trained checkpoint, the UI will open in model-unavailable mode and will not generate predictions. The app does not substitute random weights or generic ImageNet weights.

If the checkpoint is stored elsewhere, set the `MODEL_PATH` environment variable to its location.

## 3. Run locally (Ubuntu / Linux)

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU-only PyTorch wheels (the tested model code also runs on CPU)
python -m pip install --index-url https://download.pytorch.org/whl/cpu -r requirements-torch-cpu.txt

# UI and supporting libraries
python -m pip install -r requirements.txt

# Validate the real checkpoint before launching
python scripts/verify_checkpoint.py

# Start the app
streamlit run app.py
```

Open `http://localhost:8501`.

For a CUDA-enabled installation, install the matching `torch` and `torchvision` build using the official PyTorch selector, then install `requirements.txt`.

## 4. Run with Docker

1. Put `densenet121_best.pth` in `models/`.
2. Build and start:

```bash
docker compose up --build
```

3. Open `http://localhost:8501`.

The model folder is mounted read-only into the container. The checkpoint does not need to be copied into the Docker image.

To stop:

```bash
docker compose down
```

## 5. Deploy to a cloud host

Use a Docker-capable host. Build from this repository using the supplied `Dockerfile`, expose port `8501`, and set:

```text
MODEL_PATH=/app/models/densenet121_best.pth
```

Make the trained checkpoint available at that path through the host's persistent disk or a private model-artifact download step during deployment. Avoid committing a large or private checkpoint to a public Git repository.

Health endpoint:

`/_stcore/health`

## 6. Project model facts shown in the app

- Architecture: torchvision DenseNet121.
- Final layer: 1,024 input features to 6 outputs.
- Optimizer: AdamW.
- Learning rate: 0.0001.
- Weight decay: 0.0001.
- Loss: CrossEntropyLoss.
- Scheduler: StepLR, step size 3, gamma 0.1.
- Epochs: 5.
- Best checkpoint selection: validation accuracy.
- Recorded validation accuracy: 90.53%.
- Recorded held-out test accuracy: 90.91% (418 test images).

Input preprocessing mirrors the project's deterministic evaluation transform: RGB, resize to 224 × 224, tensor conversion, and ImageNet normalization with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.

Class order is taken from the notebook:
1. B12
2. Iron
3. Vitamin_A
4. Vitamin_C
5. Vitamin_D
6. Zinc

Iron and Zinc are minerals. The softmax values displayed by the app are model scores, not calibrated clinical probabilities.

## 7. Validation and tests

Run the inference smoke tests:

```bash
python -m unittest discover -s tests -v
```

The smoke test uses temporary random weights only to verify that the model architecture and preprocessing/inference code run. It is **not** an evaluation of the trained project's accuracy.

Run the checkpoint check after adding the actual trained model:

```bash
python scripts/verify_checkpoint.py models/densenet121_best.pth
```

## 8. Project structure

```text
NutrientVision_AI/
├── app.py
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── inference.py
├── models/
│   └── README.md
├── scripts/
│   └── verify_checkpoint.py
├── tests/
│   └── test_inference.py
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-torch-cpu.txt
└── README.md
```
