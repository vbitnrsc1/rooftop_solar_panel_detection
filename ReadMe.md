# Rooftop Solar Panel Detection and Segmentation

## Overview

This project focuses on detecting and segmenting rooftop solar panels in the Indian region using state-of-the-art deep learning models. The objective is to analyze satellite/aerial imagery and identify the presence of solar panels, helping in mapping solar energy installations more accurately.

## Models Used

This project was implemented by three different models to perform detection and segmentation:

### YOLOv11 (You Only Look Once v11)
- A real-time object detection model used for both training and inference.

### Faster R-CNN
- A region-based convolutional neural network designed for object detection.
- Separate notebooks for training and inference.

### Mask R-CNN
- An extension of Faster R-CNN that also provides pixel-level segmentation masks.
- Separate notebooks for training and inference.

All model training experiments were conducted using Kaggle's GPU environment, while data collection was performed using Google Colab.

## Dataset & Preprocessing

### Data Collection
- Google Maps API was used to fetch satellite images of the selected Region of Interest (ROI) at zoom level = 21.
- After downloading the images, manual filtering was performed to select images that contain solar panels.
- The selected images were then uploaded to Roboflow for annotation.

### Annotation & Preprocessing
- Annotation was performed using Roboflow's annotation tools.
- Preprocessing Steps in Roboflow:
  - Auto-Orient: Applied
  - Resize: Stretched to 640x640
- Augmentations:
  - Outputs per training example: 3
  - Exposure: Between -10% and +10%
  - Blur: Up to 2.5px
  - Noise: Up to 0.1% of pixels

## Project Structure

The project is organized with the following structure:

```
solar-panel-detection/
│
│-- data/                          # Dataset folder 
│   ├── images/                    # Sample images│
├── notebooks/
│   ├── data_collection.ipynb        # Google Colab notebook for data collection
│   ├── yolov11_training_inference.ipynb
│   ├── faster_rcnn_training.ipynb
│   ├── faster_rcnn_inference.ipynb
│   ├── mask_rcnn_training.ipynb
│   └── mask_rcnn_inference.ipynb
│
├── models/
│   ├── yolov11/            # YOLOv11 model weights
│   ├── faster_rcnn/        # Faster R-CNN model weights
│   └── mask_rcnn/          # Mask R-CNN model weights 
└── README.md            # Project documentation
```

The notebooks focus on different aspects of the project:

- `data_collection.ipynb` - Data collection using Google Maps API and preprocessing (Google Colab)
- `yolov11_training_inference.ipynb` - YOLOv11 model training and inference
- `faster_rcnn_training.ipynb` - Faster R-CNN model training
- `faster_rcnn_inference.ipynb` - Faster R-CNN model inference 
- `mask_rcnn_training.ipynb` - Mask R-CNN model training 
- `mask_rcnn_inference.ipynb` - Mask R-CNN model inference 



Credits & Acknowledgements

Kaggle for providing GPU resources.

Ultralytics, Detectron2, PyTorch for model implementations.

Open-source satellite imagery sources for dataset collection.

# test_solarrr
