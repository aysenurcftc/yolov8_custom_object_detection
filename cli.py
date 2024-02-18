import argparse

from predict import HumanFaceDetectionModel
from settings import DETECTION_MODEL


def main():
    parser = argparse.ArgumentParser(description="yolov8 custom object detection")
    parser.add_argument("input_path", help="path to input image.")
    args = parser.parse_args()

    model = HumanFaceDetectionModel()
    result = model.load_model(DETECTION_MODEL)
    
    result = model.showResult(args.input_path)


if __name__ == "__main__":
    main()
