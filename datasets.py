import numpy as np
import os
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import concurrent.futures
from tqdm import tqdm

class DataGenerator2(tf.keras.utils.Sequence):
    def __init__(self, path, y, batch_size):
        self.idx = np.arange(len(os.listdir(path)))
        self.path = path
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.idx) / float(self.batch_size)))

    def __getitem__(self, iter):
        batch_i = self.idx[iter * self.batch_size:(iter + 1) * self.batch_size]
        batch_y = self.y[iter * self.batch_size:(iter + 1) * self.batch_size]

        batch_x = []
        with concurrent.futures.ThreadPoolExecutor(10) as pool:
            futures = [
                pool.submit(np.load, os.path.join(self.path,str(i)+".npy")) 
                for i in batch_i
            ]
            concurrent.futures.wait(futures)
            batch_x = np.array([future.result() for future in futures], np.float32)

        return batch_x, batch_y

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, iter):
        batch_x = self.x[iter * self.batch_size:(iter + 1) * self.batch_size]
        batch_y = self.y[iter * self.batch_size:(iter + 1) * self.batch_size]

        return batch_x, batch_y

class Dataset():
    def __init__(
            self,
            base_model,
            name="SCUT-FBP5500",
            image_path="SCUT-FBP5500/images",
            train_path="SCUT-FBP5500/train.txt",
            test_path="SCUT-FBP5500/test.txt",
            batch_size=32,
            load_all=True,
            output_size=5,
            zero_center=True,
        ):

        self.name = name
        self.image_path = image_path
        self.batch_size = batch_size
        self.load_all = load_all
        self.output_size = output_size
        self.zero_center = zero_center

        self.train_lines = self.path_to_lines(train_path)
        self.test_lines = self.path_to_lines(test_path)
        
        self.mean = 0
        self.var = 0
        self.std = 1

        self.feature_path = None
        self.mlp_train = False
        self.base_model = base_model

    def path_to_lines(self, path):
        with open(path, "r") as f: 
            return f.readlines()

    @property
    def mlp_train(self):
        return self._mlp_train
    
    @mlp_train.setter
    def mlp_train(self, mlp_train):
        self._mlp_train = mlp_train
        if mlp_train:
            self.feature_path = os.path.join(self.feature_path,"mlp")
        elif self.feature_path:
            self.feature_path = self.feature_path_copy
                  
    @property
    def base_model(self):
        return self._base_model
    
    @base_model.setter
    def base_model(self, model):
        self._base_model = model
        self.update_feature_path()

    def check_paths(self):
        return np.array([os.path.exists(
            os.path.join(self.feature_path,name)) 
            for name in ["X_train","X_test"]
        ]).all()

    def check_lines(self, lines):
        files = []
        lines2 = []
        for l in lines:
            file = l.split(" ")[0]
            if os.path.exists(os.path.join(
                self.image_path, file
            )):
                lines2.append(l)
                files.append(file)

        return lines2, files
            
    def update_feature_path(self):
        self.feature_path = os.path.join(
            self.base_model._name,
            "features",
            self.name
        )
        if not self.mlp_train:
            self.feature_path_copy = self.feature_path
        
        if not os.path.exists(self.feature_path):
            os.makedirs(self.feature_path)

        self.create_generators() 

    def load_y(self, name):
        path = os.path.join(self.feature_path,"y_"+name+".npy")
        return np.load(path)

    def create_generators(self):
        self.train_lines, self.train_files = self.check_lines(self.train_lines)
        self.test_lines, self.test_files = self.check_lines(self.test_lines)
        
        if self.zero_center:
            print("Calculating mean and standard deviation of training set")
            self.normalise(self.train_lines)

        y_train = self.create_y(self.train_lines)
        y_test = self.create_y(self.test_lines)
       
        if self.load_all:
            t0 = time.process_time()
            X_train = self.create_x(self.train_files)
            t1 = time.process_time()
            print(f"Training set loaded in {t1-t0}s")
            self.train = DataGenerator(X_train, y_train, self.batch_size)

            t0 = time.process_time()
            X_test = self.create_x(self.test_files)
            t1 = time.process_time()
            print(f"Testing set loaded in {t1-t0}s")
            self.test = DataGenerator(X_test, y_test, self.batch_size)

            return
 
        if not self.check_paths():
            print("Creating training set")
            self.create_dataset("train", self.train_files)
            print("Creating testing set")
            self.create_dataset("test", self.test_files)

        self.train = DataGenerator2(
            os.path.join(self.feature_path,"X_train"),
            y_train,
            self.batch_size
        )
        self.test = DataGenerator2(
            os.path.join(self.feature_path,"X_test"),
            y_test,
            self.batch_size
        )
 
    def load_image(self, file):
        return img_to_array(load_img(
            os.path.join(self.image_path,file),
            target_size=(
                self.base_model.input_shape[1],
                self.base_model.input_shape[2]
            ),
            interpolation="lanczos"
        ))

    def normalise(self, lines):
        path = os.path.join(self.image_path,"zero.npy")
        if not os.path.exists(path):
            files = [l.split(" ")[0] for l in lines]
            self.mean = np.zeros(3)
            self.var = np.zeros(3)

            for file in tqdm(files):
                image = self.load_image(file)
                self.mean += np.array([
                    np.mean(image[...,0]),
                    np.mean(image[...,1]),
                    np.mean(image[...,2]),
                ])

            self.mean /= len(lines)

            for file in tqdm(files):
                image = self.load_image(file)
                self.var += np.array([
                    np.mean((image[...,0] - self.mean[0])**2),
                    np.mean((image[...,1] - self.mean[1])**2),
                    np.mean((image[...,2] - self.mean[2])**2),
                ])
            
            self.var /= len(lines)
            self.std = np.sqrt(self.var)

            np.save(path, np.array([self.mean, self.std]))

        s = np.load(path)
        self.mean = s[0]
        self.var = s[1]**2
        self.std = s[1]

    def create_y(self, lines):
        if self.output_size == 1:
            y = [l.split(" ")[6:7] for l in lines]

        elif self.output_size == 5:
            y = [l.split(" ")[1:6] for l in lines]
        
        elif self.output_size == 10:
            y = [l.split(" ")[1:11] for l in lines]

        return np.array(y, np.float32)    

    def preprocess(self, x):
        x = x[...,::-1] #convert to BGR
        x -= self.mean #zero center
        if self.base_model._name == "vgg16": x = x / self.std #if VGG16 divide by std
        x = self.base_model.predict(x, batch_size=self.batch_size, verbose=0)

        return x

    def create_x(self, files):
        with concurrent.futures.ThreadPoolExecutor(10) as pool:
            futures = [pool.submit(self.load_image, file) for file in files]
            concurrent.futures.wait(futures)

        x = np.array([future.result() for future in futures], np.float32)
        x = self.preprocess(x)
    
        return x
    
    def create_dataset(self, name, files):
        x = []
        
        path = os.path.join(self.feature_path,"X_"+name)
        if not os.path.exists(path):
            os.makedirs(path)

        x = self.create_x(files[:len(files)//2])
        for i in tqdm(range(len(files)//2)):
            np.save(os.path.join(
                path, 
                str(i)+".npy"), 
                x[i]
            )

        x = self.create_x(files[len(files)//2:])
        for i in tqdm(range(len(files)//2)):
            np.save(os.path.join(
                path, 
                str((len(files)//2)+i)+".npy"), 
                x[i]
            )