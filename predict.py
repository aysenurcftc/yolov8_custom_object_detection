from PIL import Image
from ultralytics import YOLO

from settings import *


class PandaDetectionModel:
    def __init__(self) -> None:
        self.model_path = self.load_model(model_path=DETECTION_MODEL)

    def load_model(self, model_path):
        """
        Loads a YOLO object detection model from the specified model_path.

        Parameters:
            model_path (str): The path to the YOLO model file.

        Returns:
            A YOLO object detection model.
        """
        # Load a model
        model = YOLO(model_path)
        return model

    def predictImage(self, model, input_path):
        results = model(input_path)

    def showResult(self, input_path):
        # from PIL
        im1 = Image.open(input_path)
        results = model.predict(source=im1, save=True)  # save plotted 
        return results


modelDetection = PandaDetectionModel()
model = modelDetection.load_model(DETECTION_MODEL)
results = modelDetection.showResult("./test_images/panda4.jpg")
results = model("./test_images/panda2.jpg")  # predict on an image
