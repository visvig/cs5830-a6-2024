from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model as keras_load_model
from keras.models import Sequential
from typing import List
import numpy as np
from PIL import Image
from io import BytesIO
import argparse

app = FastAPI()

# Initialize model as None at the global level
model = None

def load_model(path: str) -> Sequential:
    """Load the Keras model from a given path and handle exceptions."""
    global model
    try:
        model = keras_load_model(path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None  # model is set to None if loading fails
    return model

# global model is chosen to run unit test modules properly!
def predict_digit(data_point: List[float]) -> str:
    """Predict the digit from the image data using the model."""
    global model
    if model is None:
        raise ValueError("Model is not loaded.")
    prediction = model.predict(np.array([data_point]))
    return str(np.argmax(prediction))

def format_image(image: Image.Image) -> List[float]:
    """Resize and format the image to the correct shape for model prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = image.point(lambda x: 0 if x < 50 else 255)  # Apply threshold to remove noise
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.asarray(image)
    image_array = image_array.flatten() / 255.0  # Normalize the image
    return image_array.tolist()

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    """API endpoint to handle image file uploads and return digit predictions."""
    contents = await file.read()
    try:
        image = Image.open(BytesIO(contents))
        processed_image = format_image(image)
        digit = predict_digit(processed_image)
        return {"digit": digit}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Prediction API")
    parser.add_argument("model_path", type=str, help="Path to the saved Keras model")
    args = parser.parse_args()
    if not load_model(args.model_path):  # Check if the model was loaded successfully
        print("Failed to start the application due to model loading error.")
        exit(1)  # Exit if the model cannot be loaded
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
