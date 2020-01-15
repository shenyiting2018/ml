import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd

tf.disable_v2_behavior()


FILE_PATH = './pima-indians-diabetes.csv'

def read_csv_file(file_path):
    reader = tf.TextLineReader()

X = tf.placeholder(tf.float64, shape = (), name = 'X')
t = tf.placeholder(tf.float64, shape = (), name = 't')
n = tf.placeholder(tf.float64, name = 'n')
XT = tf.transpose(X)
w = tf.matmul(tf.matrix_inverse(tf.matmul(XT, X), XT), t)

y = tf.matmul(X, w)

