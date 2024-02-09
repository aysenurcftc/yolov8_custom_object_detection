import argparse

from predict import PandaDetectionModel
from settings import DETECTION_MODEL


def main():
    parser = argparse.ArgumentParser(description="detection")
    parser.add_argument("input_path", help="path to input image.")
    args = parser.parse_args()

    model = PandaDetectionModel()
    result = model.load_model(DETECTION_MODEL)
    result = model.showResult(args.input_path)


if __name__ == "__main__":
    main()
