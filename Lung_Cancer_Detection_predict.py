import tensorflow as tf
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
args = parser.parse_args()

model = tf.keras.models.load_model('lung_model.h5')
img = cv2.imread(args.image)
img = cv2.resize(img, (150, 150))
img = np.expand_dims(img / 255.0, axis=0)
pred = model.predict(img)
classes = ['benign', 'malignant', 'normal']
print(f"Prediction: {classes[np.argmax(pred)]}")