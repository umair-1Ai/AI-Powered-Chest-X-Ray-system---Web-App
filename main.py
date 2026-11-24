"""
AXDS backend - FastAPI + TensorFlow Lite.

Model files (put next to this file, or override with env vars):
  efficientnetb3_gradcam.tflite     -> predictions + Grad-CAM (preferred)
  efficientnetb3_covid19_model.tflite -> predictions only (fallback, no heat-map)

The Grad-CAM model is exported by the notebook cell `export_gradcam_tflite()`.
It returns BOTH the softmax probabilities and a normalised CAM from one forward
pass, so the label and the heat-map always describe the same prediction.
IMPORTANT: it must come from the same training run as the classifier.
"""
import io
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
GRADCAM_PATH = Path(os.getenv("GRADCAM_MODEL_PATH", BASE_DIR / "efficientnetb3_gradcam.tflite"))
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_MODEL_PATH", BASE_DIR / "efficientnetb3_covid19_model.tflite"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", BASE_DIR / "labels.json"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs

app = FastAPI(title="AXDS - AI Chest X-Ray API")

# allow_credentials must stay False when origins is "*" (browsers reject the combination)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)


class TFLiteModel:
    """Thread-safe wrapper (a TFLite Interpreter must not be used concurrently)."""

    def __init__(self, path: Path):
        self.path = path
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()
        _, self.h, self.w, _ = [int(d) for d in self.input["shape"]]
        self.lock = threading.Lock()

        # Output order is not guaranteed after conversion -> identify by rank.
        self.probs_idx = next(i for i, o in enumerate(self.outputs) if len(o["shape"]) == 2)
        self.cam_idx = next((i for i, o in enumerate(self.outputs) if len(o["shape"]) == 3), None)
        self.has_cam = self.cam_idx is not None

    def run(self, batch: np.ndarray):
        with self.lock:
            self.interpreter.set_tensor(self.input["index"], batch)
            self.interpreter.invoke()
            probs = self.interpreter.get_tensor(self.outputs[self.probs_idx]["index"]).copy()
            cam = None
            if self.has_cam:
                cam = self.interpreter.get_tensor(self.outputs[self.cam_idx]["index"]).copy()
        return probs, cam


if GRADCAM_PATH.exists():
    model = TFLiteModel(GRADCAM_PATH)
elif CLASSIFIER_PATH.exists():
    model = TFLiteModel(CLASSIFIER_PATH)
else:
    raise RuntimeError(f"No model found. Expected {GRADCAM_PATH.name} or {CLASSIFIER_PATH.name} in {BASE_DIR}")

if len(label_map) != model.outputs[model.probs_idx]["shape"][1]:
    raise RuntimeError("labels.json does not match the model's number of classes")


def preprocess_image(data: bytes) -> np.ndarray:
    """Decode -> RGB -> resize exactly like the training notebook.

    Training used cv2.resize (bilinear) on RGB pixels with raw 0-255 values
    (EfficientNet rescales inside the model). PIL's default resize is bicubic,
    which shifts the probabilities slightly, so cv2 is used here instead.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(status_code=400, detail="Could not read the file as an image (PNG/JPG expected).")
    arr = np.asarray(image, dtype=np.uint8)
    arr = cv2.resize(arr, (model.w, model.h), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(arr.astype(np.float32), axis=0)


INDEX_PATH = BASE_DIR / "index.html"


# Only index.html is exposed (not the whole folder), so model files and source stay private.
@app.get("/", include_in_schema=False)
def index():
    if INDEX_PATH.exists():
        return FileResponse(INDEX_PATH, media_type="text/html")
    return {"message": "COVID Detection API Running", "heatmap": model.has_cam}


@app.get("/api")
def api_info():
    return {"message": "COVID Detection API Running", "heatmap": model.has_cam}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model.path.name,
        "input_size": [model.h, model.w],
        "heatmap": model.has_cam,
        "classes": label_map,
    }


# Plain `def` -> FastAPI runs it in a worker thread, so inference never blocks the event loop.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    batch = preprocess_image(data)

    start = time.perf_counter()
    probs, cam = model.run(batch)
    inference_ms = round((time.perf_counter() - start) * 1000, 2)

    scores = probs[0]
    pred_index = int(np.argmax(scores))
    prediction = label_map[str(pred_index)]
    all_scores = {label_map[str(i)]: float(scores[i]) for i in range(len(label_map))}

    heatmap = None
    if cam is not None:
        grid = np.clip(cam[0], 0.0, None)
        peak = float(grid.max())
        if peak > 1e-12:  # normalise defensively (the model already does)
            grid = grid / peak
        heatmap = {
            "method": "grad-cam",
            "class": prediction,  # the CAM explains exactly this prediction
            "width": int(grid.shape[1]),
            "height": int(grid.shape[0]),
            "data": [round(float(v), 4) for v in grid.reshape(-1)],  # row-major, values 0..1
        }

    return {
        "prediction": prediction,
        "confidence": round(float(scores[pred_index]) * 100, 2),
        "all_scores": all_scores,
        "inference_ms": inference_ms,
        "heatmap": heatmap,
    }