from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "yolo11n_v1_100epoch.pt"
CONFIDENCE = 0.40
IMAGE_SIZE = 640

# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="Floating Waste Detection API",
    description="YOLO11n V1 floating-waste detection backend",
    version="1.0.0",
)

# Load the model once when the server starts.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Place yolo11n_v1_100epoch.pt inside the backend folder."
    )

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully")
print("Model:", MODEL_PATH)
print("Classes:", model.names)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "model": "YOLO11 Floating Waste V1",
        "confidence": CONFIDENCE,
        "classes": list(model.names.values()),
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class": model.names[class_id],
                    "confidence": round(confidence, 3),
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                }
            )

    return {
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
    }
