import tensorflow as tf
import numpy as np
import os
import sys

from datasets import Dataset
import losses
import cnns
from models import Model

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.random.set_seed(1234)

parent_name = sys.argv[1]
weights = sys.argv[2]
name = sys.argv[3]
k = sys.argv[4]
output_size = int(sys.argv[5])

#output_size = 5
#parent_dataset = "mediapipe"
#k = 1
#weights = "vggface"
#dataset = "SCUT-FBP5500"

print(dataset,weights,name,k)
metrics = [
    losses.MeanAbsoluteError(output_size=output_size),
    losses.RootMeanSquaredError(output_size=output_size),
    losses.PearsonCorrelation(output_size=output_size),
]

os.chdir("C:/Users/ugail/Documents/paperV2")
cnn = cnns.ResNet50(weights=weights)
cnn.construct()

dataset = Dataset(
    cnn.base,
    name=name,
    image_path=os.path.join(parent_name,name),
    train_path=os.path.join(parent_name,"train.txt"),
    test_path=os.path.join(parent_name,"test.txt"),
    load_all=True,
    output_size=output_size,
    batch_size=64,
)
model = Model(
    cnn,
    name=name,
    parent_name=parent_name,
    loss="categorical_crossentropy", 
    dataset=dataset,
    folder=k,
    output_size=output_size,
)
model.construct(
    activation="softmax",
    supress=True,
)
model.compile(
    metrics=metrics,
    learning_rate=0.0001
)

model.train(patience=35)
model.predict(save_name="y_pred_test", test_set=False)
model.predict(save_name="y_pred_train", test_set=True)