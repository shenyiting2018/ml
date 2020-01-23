import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from sklearn import datasets
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from tensorflow.python.framework import ops
import pandas as pd

tf.disable_v2_behavior()

ops.reset_default_graph()

sess = tf.Session()

FILE_PATH = './Iris.csv'


dataset_initial = pd.read_csv(FILE_PATH)
num_sample = dataset_initial.shape[0]
num_attr = dataset_initial.shape[1] - 2
num_cluster = 3

df = pd.DataFrame(dataset_initial)
dataset = np.array(df.drop(['Sample Index', 'outcome(Cluster Index)'], axis=1))

#iris = datasets.load_iris()
iris = dataset
#iris2 = datasets.load_iris().data
import ipdb; ipdb.set_trace()
num_pts = len(iris)
num_feats = len(iris[0])

# Set k-means parameters
# There are 3 types of iris flowers, see if we can predict them
k = 3
generations = 55

data_points = tf.Variable(iris)
cluster_labels = tf.Variable(tf.zeros([num_pts], dtype=tf.int64))

# Randomly choose starting points
rand_starts = np.array([iris[np.random.choice(len(iris))] for _ in range(k)])

centroids = tf.Variable(rand_starts)

# In order to calculate the distance between every data point and every centroid, we
#  repeat the centroids into a (num_points) by k matrix.
centroid_matrix = tf.reshape(tf.tile(centroids, [num_pts, 1]), [num_pts, k, num_feats])
# Then we reshape the data points into k (3) repeats
point_matrix = tf.reshape(tf.tile(data_points, [1, k]), [num_pts, k, num_feats])
distances = tf.reduce_sum(tf.square(point_matrix - centroid_matrix), axis=2)
obj_j = tf.reduce_sum(distances)

# Find the group it belongs to with tf.argmin()
centroid_group = tf.argmin(distances, 1)


# Find the group average
def data_group_avg(group_ids, data):
    # Sum each group
    sum_total = tf.unsorted_segment_sum(data, group_ids, 3)
    # Count each group
    num_total = tf.unsorted_segment_sum(tf.ones_like(data), group_ids, 3)
    # Calculate average
    avg_by_group = sum_total / num_total
    return (avg_by_group)


means = data_group_avg(centroid_group, data_points)

update = tf.group(centroids.assign(means), cluster_labels.assign(centroid_group))

init = tf.global_variables_initializer()

sess.run(init)

distances_values = []
distances_values.append(0)
for i in range(generations):
    print('Calculating gen {}, out of {}.'.format(i, generations))
    obj_j_value, distances_value, _, centroid_group_count = sess.run([obj_j, distances, update, centroid_group])
    distances_values.append(obj_j_value)
    group_count = []
    for ix in range(k):
        group_count.append(np.sum(centroid_group_count == ix))
    print('Group counts: {}'.format(group_count))
