# Object-Detection-Using-Python
This project implements an object detection system using the YOLOv3 pre-trained model. The goal is to localize and classify objects within an image or through a real-time video feed using the power of computer vision.

## Overview
Object detection is a crucial computer vision task that involves both identifying the presence of objects in an image and precisely localizing them. This project is based on the YOLO (You Only Look Once) architecture, which allows for unified, real-time object detection. The core idea of YOLO is to predict bounding boxes and class probabilities for objects directly in one pass through the neural network.

## Getting Started

1. Clone this repository to your local machine using:
   ```
   git clone https://github.com/RaviVishwakarma11/Object-Detection.git
   ```

2. Download the YOLOv3 pre-trained weights (yolov3.weights) from [here](https://pjreddie.com/darknet/yolo/) and place it in the project directory.

3. Install the required Python libraries that mention in data.py file code.

4. Run the object detection script using:
   ```
   python data.py
   ```

## Features
- Real-time object detection using the YOLOv3 model.
- Detection of multiple objects within an image or video stream.
- Bounding box visualization around detected objects.
- Class prediction for each detected object.
- Integration with OpenCV for capturing real-time video feed and object placement in front of a camera.
- Support for linking images to perform object detection on them.

## Usage

- To perform real-time object detection using your computer's camera, run the script and the camera feed will display objects with bounding boxes and class labels in real-time.
- To perform object detection on a specific image, use the image link functionality or specify the image path in the script.
