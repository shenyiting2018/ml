import matplotlib
import np as np
import pandas as pd
import tensorflow.compat.v1 as tf
import numpy as np

print(tf.__version__)


import matplotlib.pyplot as plt
from datetime import datetime


tf.disable_v2_behavior()
# tf.executing_eagerly()

FILE_PATH = './Iris.csv'


dataset_initial = pd.read_csv(FILE_PATH)
num_sample = dataset_initial.shape[0]
num_attr = dataset_initial.shape[1] - 2
num_cluster = 3

df = pd.DataFrame(dataset_initial)
dataset = np.array(df.drop(['Sample Index', 'outcome(Cluster Index)'], axis=1))

EPSILON = 0.00005

points = tf.Variable(initial_value=dataset, name='points')
centroids = tf.Variable(tf.slice(tf.random.shuffle(points), [0, 0], [num_cluster, num_attr]))
points_expanded = tf.expand_dims(dataset, 0)
centroids_expanded = tf.expand_dims(centroids, 1)
distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
obj_j = tf.reduce_sum(distances)
assignments = tf.argmin(distances, 0)

# calculate update centroids
means = []
for cluster in range(num_cluster):
    means.append(
        tf.compat.v1.reduce_mean(
            tf.gather(
                dataset,
                tf.reshape(
                    tf.where(
                        tf.equal(assignments, cluster)
                    ),
                    [1,-1])
               ),
            reduction_indices=[1],
        ),
    )


new_centroids = tf.concat(means, 0)
update_centroids = tf.compat.v1.assign(centroids, new_centroids)

# Init values
init = tf.compat.v1.global_variables_initializer()
iteration_n = 100
distances_values = []
distances_values.append(0)
# Run training
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for step in range(1000):
        [update_centroids_values, centroid_values, points_values, assignment_values, obj_j_value]\
            = sess.run(
                [
                    update_centroids,
                    centroids,
                    points,
                    assignments,
                    obj_j,
                ],
            )
        distances_values.append(obj_j_value)
        diff = abs(distances_values[step + 1] - distances_values[step])
        print("centroids: {}".format(centroid_values))
        print("distance: {}".format(obj_j_value))
        print("diff:{}".format(diff))
        if (step > 100 and diff < EPSILON):
            break



import ipdb; ipdb.set_trace()
