import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Flatten
import numpy as np

def wrap(loss, name, mode, output_size):
    name = name + ("" if mode == "mean" else "_" + mode)
 
    def wrapped_loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        if output_size == 5:
            y_true = tf.tensordot(y_true, tf.range(1,6,dtype=tf.float32), axes=1)
            y_pred = tf.tensordot(y_pred, tf.range(1,6,dtype=tf.float32), axes=1)
        elif output_size == 10:
            y_true = tf.tensordot(y_true, tf.range(1,11,dtype=tf.float32), axes=1)
            y_pred = tf.tensordot(y_pred, tf.range(1,11,dtype=tf.float32), axes=1)

        return loss(y_true, y_pred)

    exec(f"def {name}(y_true, y_pred): return wrapped_loss(y_true, y_pred)",
        {"wrapped_loss" : wrapped_loss}, locals()
    )
    return locals()[name]

def MeanAbsoluteError(mode="mean", output_size=5):
    loss = lambda y_true, y_pred : tf.reduce_mean(tf.abs(y_true - y_pred))
    name = "mean_absolute_error"
    return wrap(loss, name, mode, output_size)

def RootMeanSquaredError(mode="mean", output_size=5):
    loss = lambda y_true, y_pred : tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    name = "root_mean_squared_error"
    return wrap(loss, name, mode, output_size)

def MeanSquaredError(mode="mean", output_size=5):
    loss = lambda y_true, y_pred : tf.reduce_mean(tf.square(y_true - y_pred))
    name = "mean_squared_error"
    return wrap(loss, name, mode, output_size)

def PearsonCorrelation(mode="mean", output_size=5):
    loss = lambda y_true, y_pred : tfp.stats.correlation(Flatten()(y_true), Flatten()(y_pred))
    name = "correlation"
    return wrap(loss, name, mode, output_size)