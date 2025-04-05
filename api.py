from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
import json
import os

# Load your fine-tuned model (ncnn format is loaded like any other)
model = YOLO('yolo11x-seg-ft_ncnn_model')

# Define the API app
app = FastAPI()

# Request format
class InferenceRequest(BaseModel):
    image_path: str

# Inference endpoint
@app.post("/infer")
def run_inference(request: InferenceRequest):
    image_path = request.image_path

    if not os.path.isfile(image_path):
        raise HTTPException(status_code=400, detail="Image path not found.")

    # Run inference
    results = model(image_path)

    # ⚠️ Convert result to JSON-safe format before returning
    return json.loads(results[0].tojson())