
from fastapi import FastAPI, File, UploadFile
from predict import PandaDetectionModel
import os
import uuid

from settings import DETECTION_MODEL

app = FastAPI()
model = PandaDetectionModel()
temp = "api_images"
os.makedirs(temp, exist_ok=True)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1]
        if file_ext not in {".jpg", ".jpeg", ".png"}:
            return {"error" : "Uploaded file must be in JPG, JPEG or PNG format."}
        
        filename_base = str(uuid.uuid4())
        filename = filename_base + file_ext
        destination_path = os.path.join(temp, filename)
        output_path = os.path.join(temp, "output" + filename_base + ".png")
        
        with open(destination_path, "wb") as image_data:
            image_data.write(file.file.read())
            
            
        model.load_model(DETECTION_MODEL)
        model.showResult(destination_path)
        
        return {"OK": "Image has been saved."}
        
    except Exception as e:
        return {"error" : str(e)}