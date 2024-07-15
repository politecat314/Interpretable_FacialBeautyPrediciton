import tensorflow as tf
import os
import math
import losses
import numpy as np
import keras_tuner as kt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import cv2

class Expectation(tf.keras.layers.Layer):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.tensordot(inputs, tf.range(1,self.output_size+1,dtype=tf.float32), axes=1)

class Model():
    def __init__(
            self, 
            cnn, 
            loss="categorical_crossentropy",
            parent_name="SCUT-FBP5500",
            name=None, 
            dataset=None, 
            folder=0,
            output_size=5,
            load_weights=False,
        ):

        self.cnn = cnn
        self.loss = loss
        self.dataset = dataset
        self.output_size = output_size
        self.load_weights = load_weights
        self.parent_name = parent_name
        self.name = name

        if self.dataset and not name:
            self.name = self.dataset.name
        self.mlp_path = os.path.join(self.cnn.name, f"mlp{self.output_size}.hdf5")

        self.directory = os.path.join(
            self.cnn.name,
            str(folder),
            self.loss,
            self.parent_name,
            self.name
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.weights_path = os.path.join(self.directory,"weights.h5")

    def construct(
            self,
            activation="softmax",
            supress=True
        ):

        tf.keras.backend.clear_session()
        self.cnn.construct()

        #MLP
        inputs = tf.keras.layers.Input(
            self.cnn.train.output_shape[1:]
        )
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(
            128, kernel_regularizer=tf.keras.regularizers.L2()
        )(x)
        x = tf.keras.layers.PReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(
            self.output_size, activation=activation,
        )(x)
        
        mlp = tf.keras.Model(inputs, outputs)
        mlp._name = "MLP"
        if not supress: mlp.summary()

        if not os.path.exists(self.mlp_path):
            self.dataset.mlp_train = True
            self.dataset.base_model = self.cnn.full
            self.model = mlp
            self.compile(
                metrics=[losses.RootMeanSquaredError(output_size=self.output_size)]
            )
            self.train(monitor="val_root_mean_squared_error", epochs=300, patience=30)
            self.dataset.mlp_train = False
            self.dataset.base_model = self.cnn.base
        
        mlp.load_weights(self.mlp_path)

        inputs = tf.keras.layers.Input(
            self.cnn.base.output_shape[1:]
        )
        x = inputs
        x = self.cnn.train(x)
        if not supress: self.cnn.train.summary()
        outputs = mlp(x)

        self.model = tf.keras.Model(inputs, outputs)

        if os.path.exists(self.weights_path) and self.load_weights:
            self.model.load_weights(self.weights_path)
            if self.output_size > 1:
                self.full_model = tf.keras.Sequential([
                    self.cnn.base,
                    self.model,
                    #Expectation(self.output_size)
                ])
            else:
                self.full_model = tf.keras.Sequential([
                    self.cnn.base,
                    self.model
                ])

    def compile(self, metrics=None, learning_rate=0.0001):
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            learning_rate,
            decay_steps=2500,
            decay_rate=1,
            staircase=False
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=self.loss,
            metrics=metrics
        )

    def train(
            self,
            callbacks=None,
            epochs=500, 
            patience=30, 
            verbose=1,
            monitor="val_mean_absolute_error", 
        ):

        if monitor == "val_correlation":
            mode = "max"
        else: 
            mode = "min"

        path = self.weights_path 
        if not os.path.exists(self.mlp_path):
            path = self.mlp_path

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )

        if callbacks:
            callbacks += [checkpoint]
        else:
            callbacks = [checkpoint]

        if patience:
            stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                mode=mode,
            )

            callbacks += [stopping]

        self.model.fit(
            self.dataset.train,
            epochs=epochs,
            validation_data=self.dataset.test,
            callbacks=callbacks,
            verbose=verbose
        )

        self.model.load_weights(path)
        
    def predict(
            self,
            path=None,
            save_name="y_pred_test",
            test_set=True,
            metrics=False,
        ):

        if test_set:
            self.model.evaluate(self.dataset.test)
            y_pred = self.model.predict(self.dataset.test)
            y_test = self.dataset.test.y
        else:
            self.model.evaluate(self.dataset.train)
            y_pred = self.model.predict(self.dataset.train)
            y_test = self.dataset.train.y

        if not path:
            path = self.directory

        if not metrics:
            np.save(os.path.join(path, save_name+".npy"), y_pred)
        
        if self.output_size == 5:
            y_test = y_test@np.arange(1,6)
            y_pred = y_pred@np.arange(1,6)
        elif self.output_size == 10:
            y_test = y_test@np.arange(1,11)
            y_pred = y_pred@np.arange(1,11)
        else:
            y_test = y_test[...,0]
            y_pred = y_pred[...,0]

        print("-------------------------")
        print(np.sqrt(mean_squared_error(y_test, y_pred)))
        print(mean_absolute_error(y_test, y_pred))
        print(pearsonr(y_test, y_pred)[0])
        print("-------------------------")

        if not metrics:
            return

        np.save(
            os.path.join(path, save_name+".npy"),
            np.array([
                np.sqrt(mean_squared_error(y_test, y_pred)),
                mean_absolute_error(y_test, y_pred),
                pearsonr(y_test, y_pred)[0],
            ])
        )

    def preprocess(self, x):
        x = np.array(x[...,::-1], np.float32)
        x -= np.load(os.path.join(
            self.parent_name,
            self.name,
            "zero.npy"
        ))[0]
        return x

    def inference(self, path):
        image = cv2.imread(path)[...,::-1]
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LANCZOS4)
        image = self.preprocess(image)
        features = self.cnn.base.predict(np.array([image]), verbose=0)
        return self.model.predict(features, verbose=0)[0]

    def batch_inference(self, x, preprocess=False, batch_size=128):
        if preprocess:
            x = self.preprocess(x)
            x = self.cnn.base.predict(x, batch_size=batch_size)
        return  self.model.predict(x, batch_size=batch_size)








    """
    def search(
            self,
            max_epochs=275,
            objective="val_root_mean_squared_error",
            mode="min",
            factor=5,
            hyperband_iterations=1,
            directory="./tuner/",
        ):

        self.tuner = tuner = kt.Hyperband(
            hypermodel=self.mlp,
            objective=kt.Objective(objective, direction=mode),
            max_epochs=max_epochs,
            hyperband_iterations=hyperband_iterations,
            factor=factor,
            max_retries_per_trial=1,
            overwrite=True,
            directory=directory,
            project_name="yesnowfam",
        )
        self.tuner.search_space_summary()

        stopping = tf.keras.callbacks.EarlyStopping(
            monitor=objective,
            patience=30,
            mode=mode,
        )

        tensorboard = tf.keras.callbacks.TensorBoard(directory)
        tuner.search(
            self.dataset.train,
            epochs=(
                hyperband_iterations
                *int(max_epochs*(math.log(max_epochs, factor)**2))
            ),
            validation_data=self.dataset.test,
            callbacks=[stopping, tensorboard]
        )

        self.tuner.results_summary()

    def mlp(self, hp):
        #units = hp.Int(
            #"units",
            #min_value=32,
            #max_value=512,
            #step=32
        #)
        #dropout = hp.Float(
            #"dropout",
            #min_value=0.2,
            #max_value=0.8,
            #step=0.05
        #)
        #learning_rate = hp.Choice(
            #"learning_rate",
            #values=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
        #)
        #loss = hp.Choice(
            #"loss",
            #["mean_squared_error", "mean_absolute_error", "huber", "log_cosh"]
        #)

        inputs = tf.keras.layers.Input(self.cnn.train.output_shape[1:])
        x = tf.keras.layers.Flatten()(inputs)

        #if hp.Boolean("bool"):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        outputs = tf.keras.layers.Dense(2)(x)
        mlp = tf.keras.Model(inputs, outputs)

        #inputs = tf.keras.layers.Input(self.cnn.base.output_shape[1:])
        #x = self.cnn.train(inputs)
        #outputs = mlp(x)
        #self.model = tf.keras.Model(inputs, outputs)

        mlp.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
            loss="mean_squared_error",
            metrics=[losses.RootMeanSquaredError(), losses.PearsonCorrelation()]
        )

        return mlp
    """