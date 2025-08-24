# Lung Cancer Detection System

CNN-based system to classify lung cancer from CT scans using TensorFlow/Keras.

## Setup
1. Download dataset (e.g., IQ-OTH/NCCD from Kaggle) and place in `data/` with subfolders: benign, malignant, normal.
2. Install: `pip install -r requirements.txt`
3. Train: `python train.py`
4. Predict: `python predict.py --image path/to/image.jpg`

## Dependencies
- tensorflow, keras, opencv-python, numpy, scikit-learn