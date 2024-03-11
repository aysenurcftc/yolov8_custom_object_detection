
# Yolov8-Custom-Object-Detetction

This repo detects panda with YOLOv8 from ultralytics

## Usage/Examples

### CLI Usage
```bash
usage: cli.py [-h] input_path

yolov8 custom object detection

positional arguments:
  input_path  path to input image.

options:
  -h, --help  show this help message and exit
```
### API Usage

```
http://127.0.0.1:8041/predict
```
## Installation

Install project with pip

```bash
pip install -r requirements.txt
```

## Deployment
To deploy this project run
```bash
docker build -t yolo_det .
docker run -d -p 8041:8041 yolo_det
```

    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`IMG_API_KEY`


## License

[MIT](https://choosealicense.com/licenses/mit/)

