import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing()

N = housing.data.shape[0]

housing_data_plus_bias = np.c_[np.ones((N, 1)), housing.data]

target_val = housing.target.reshape(-1, 1)

# import ipdb; ipdb.set_trace();

X_train, X_test, t_train, t_test = \
	train_test_split(housing_data_plus_bias, target_val, test_size=0.2, random_state=42)

n_train, m = X_train.shape
n_test, m = X_test.shape

X = tf.placeholder(tf.float64, shape = (None, m), name = 'X')
t = tf.placeholder(tf.float64, shape = (None, 1), name = 't')
n = tf.placeholder(tf.float64, name = 'n')
XT = tf.transpose(X)
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)

y = tf.matmul(X, w)

MSE = tf.div(tf.matmul(tf.transpose(y - t), y - t), n)

w_star = tf.placeholder(tf.float64, shape = (m, 1), name='w_star')
y_test = tf.matmul(X, w_star)

MSE_test = tf.div(tf.matmul(tf.transpose(y_test-t), y_test-t), n)

with tf.Session() as sess:
	MSE_train_val, w_val = \
	sess.run([MSE, w], feed_dict={X: X_train, t: t_train, n: n_train})

	MSE_test_val = \
	sess.run([MSE_test], feed_dict={X: X_test, t: t_test, n: n_test, w_star: w_val})
