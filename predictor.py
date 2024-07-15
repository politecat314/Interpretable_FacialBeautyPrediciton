import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from datasets import Dataset
import losses
import cnns
from models import Model

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tqdm import tqdm
import concurrent.futures
import sys


parent_name = sys.argv[1]
output_size = int(sys.argv[2])
feature_name = sys.argv[3]
weights_name = sys.argv[4]
metrics = int(sys.argv[5])
permutation = int(sys.argv[6])
save_name_test = sys.argv[7]
save_name_train = sys.argv[8]

"""
parent_name = "SCUT-FBP5500"
output_size = int(5)
feature_name = "mediapipe"
weights_name = "mediapipe"
metrics = int("1")
permutation = int("1")
save_name_test = "s_test"
save_name_train = "s_train"
"""

os.chdir("C:/Users/ugail/Documents/paperV2")
metrics = [
    losses.MeanAbsoluteError(output_size=output_size),
    losses.RootMeanSquaredError(output_size=output_size),
    losses.PearsonCorrelation(output_size=output_size),
]

cnn = cnns.ResNet50(weights="vggface")
cnn.construct()
dataset = Dataset(
    cnn.base,
    name=feature_name,
    image_path=os.path.join(parent_name,feature_name),
    train_path=os.path.join(parent_name,"train.txt"),
    test_path=os.path.join(parent_name,"test.txt"),
    load_all=True,
    zero_center=True,
    output_size=output_size,
    batch_size=64,
)

resnet = Model(
    cnn,
    dataset=dataset,
    parent_name=parent_name,
    name=weights_name,
    output_size=output_size,
    load_weights=True
)
resnet.construct(activation="softmax")
resnet.compile(metrics=metrics, learning_rate=0.0001)

path = None
if permutation:
    path = os.path.join(
        "permutation",
        "vggface",
        parent_name,
        feature_name
    )

    if not os.path.exists(path):
        os.makedirs(path)

resnet.predict(path=path, test_set=True, save_name=save_name_test)
resnet.predict(path=path, test_set=False, save_name=save_name_train)


