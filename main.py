from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

# Load the trained model
MODEL_PATH = "model2.h5"  # Ensure this file exists in the working directory
model = load_model(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))  # Resize to match model input
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Allow requests from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")  # Ensure RGB format
        img_array = preprocess_image(img)
        
        prediction = model.predict(img_array)[0][0]  # Get the probability
        print(prediction)
        label = "Stress" if prediction > 0.5 else "No Stress"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)
        
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

# Run the app if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
