import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from datasets import Dataset
import losses
import cnns
from models import Model
from cropping import align, crop_face, detect_landmarks, crop_feature, draw_boxes

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tqdm import tqdm
import concurrent.futures
import sys
import saliency.core as saliency
import tensorflow as tf
import cv2
import lime
from lime import lime_image
import matplotlib.pyplot as plt

names = [
    "nose",
    "lips",
    "eyes",
    "left_eye",
    "right_eye",
    "cheeks",
    "right_cheek",
    "left_cheek",
    "chin",
    "eyebrows",
    "left_eyebrow",
    "right_eyebrow",
]

dataset = "MEBeauty"
output_size = 10
os.chdir("C:/Users/ugail/Documents/paperV2")

cnn = cnns.ResNet50(weights="vggface")
cnn.construct()
resnet = Model(
    cnn, 
    parent_name=dataset,
    name="mediapipe", 
    loss="categorical_crossentropy", 
    output_size=output_size,
    load_weights=True
)
resnet.construct(activation="softmax")

with open(f"{dataset}/train.txt", "r") as f:
    lines = f.readlines()
    y_train = np.array([l.split(" ")[6:7] for l in lines], np.float32)
    train_files = [l.split(" ")[0] for l in lines]

with open(f"{dataset}/test.txt", "r") as f:
    lines = f.readlines()
    y_test = np.array([l.split(" ")[6:7] for l in lines], np.float32)
    test_files = [l.split(" ")[0] for l in lines]

image = align(os.path.join(dataset,"images",test_files[0]))
m1, m2 = crop_face(image)
image_resized = cv2.resize(
    image[m1[1]:m2[1],m1[0]:m2[0]], 
    (224,224), 
    interpolation=cv2.INTER_LANCZOS4
)[...,::-1]
image = resnet.preprocess(image_resized)

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image, resnet.full_model.predict, num_samples=10)
print(explanation.top_labels)
ind =  explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()