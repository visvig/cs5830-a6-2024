from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from typing import List
import numpy as np
from PIL import Image
from io import BytesIO
import argparse

app = FastAPI()

model = None  # Initialize model as None

def load_model(path: str) -> Sequential:
    """Attempt to load the model and handle exceptions if it fails."""
    global model
    try:
        model = keras_load_model(path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    return model

# global model is chosen to run unit test modules properly!
def predict_digit(data_point: List[float]) -> str:
    global model
    if model is None:
        raise ValueError("Model is not loaded.")
    prediction = model.predict(np.array([data_point]))
    return str(np.argmax(prediction))

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
        image = image.convert("L")
        image_array = np.asarray(image)
        flattened_image_array = image_array.flatten() / 255.0
        digit = predict_digit(flattened_image_array.tolist())
        return {"digit": digit}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Prediction API")
    parser.add_argument("model_path", type=str, help="Path to the saved Keras model")
    args = parser.parse_args()
    load_model(args.model_path)  # model is loaded here
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
