# Pothole Detection ML Model

This directory contains the machine learning components for the Pothole Detection System.

## How to get a trained model

The system currently uses an OpenCV-based heuristic as a fallback when no trained CNN model is found. To enable high-accuracy CNN detection, you need to provide a trained Keras model (`.h5` file).

### 1. Obtain a Dataset
You can find several pothole datasets on Kaggle:
- [Pothole Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/viren78/pothole-detection-dataset)
- [Pothole and Non-Pothole Images](https://www.kaggle.com/datasets/chithrababu/pothole-and-plain-road-images)

### 2. Organize the Dataset
Organize your images into the following folder structure:
```text
dataset/
    train/
        no_pothole/      (images of normal roads)
        small_pothole/   (images of small potholes)
        medium_pothole/  (images of medium potholes)
        large_pothole/   (images of large potholes)
    val/                 (optional validation set with same subfolders)
        ...
```

*Note: If your dataset only has "pothole" and "no_pothole", you can place all potholes in one of the categories or modify the training script to match your classes.*

### 3. Train the Model
Run the provided training script:
```bash
cd backend/app/ml
python train_model.py --dataset /path/to/your/dataset --epochs 20
```

This will:
1. Create a model using **MobileNetV2** (Transfer Learning).
2. Train the top layers first.
3. Fine-tune the base model.
4. Save the result to `models/pothole_model.h5`.

### 4. Automatic Loading
Once the model is saved at `backend/app/ml/models/pothole_model.h5`, the backend will automatically load it on the next startup and use it for all detections instead of the OpenCV fallback.

## Model Details
- **Architecture**: MobileNetV2 (weights='imagenet') + Custom Dense Layers.
- **Input Size**: 224x224 RGB.
- **Classes**: `no_pothole`, `small_pothole`, `medium_pothole`, `large_pothole`.
- **Output**: Softmax probabilities for each class.
