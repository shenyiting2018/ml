import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
import pandas as pd
import matplotlib


FILE_PATH = './Iris.csv'

# Avoid v2 behaviours
tf.disable_v2_behavior()
# Reset canvas
ops.reset_default_graph()
axis_x = []
axis_y = []

# Define metadata
num_cluster = 3
EPSILON = 0.00005


# Initialize dataset
dataset_initial = pd.read_csv(FILE_PATH)
df = pd.DataFrame(dataset_initial)
dataset = np.array(df.drop(['Sample Index', 'outcome(Cluster Index)'], axis=1))


# Get dimensions of data
num_samples = len(dataset)
dimensions = len(dataset[0])
num_clusters = 3


# Define formula
# define data points
data_points = tf.Variable(dataset)
cluster_labels = tf.Variable(tf.zeros([num_samples], dtype=tf.int64))
# Define initial centroids
initial_centroids = np.array(
    [dataset[np.random.choice(len(dataset))] for _ in range(num_clusters)]
)
centroids = tf.Variable(initial_centroids)
# Do some reshape to apply subtraction
expanded_centroids = tf.reshape(tf.tile(centroids, [num_samples, 1]), [num_samples, num_clusters, dimensions])
expanded_points = tf.reshape(tf.tile(data_points, [1, num_clusters]), [num_samples, num_clusters, dimensions])

# Calculate distance
distances = tf.reduce_sum(tf.square(expanded_points - expanded_centroids), axis=2)
# Assign a cluster to each point
assignments = tf.argmin(distances, 1)


# Update new cluster
def data_group_avg(assignments, data):
    sum_total = tf.unsorted_segment_sum(data, assignments, 3)
    num_total = tf.unsorted_segment_sum(tf.ones_like(data), assignments, 3)
    avg_by_group = sum_total / num_total
    return avg_by_group

means = data_group_avg(assignments, data_points)
update = tf.group(centroids.assign(means), cluster_labels.assign(assignments))


# Calculate objective
def calculate_objective(centroids, assignments, points):
    obj_list = []
    for idx in range(num_samples):
        ass = assignments[idx]
        m = centroids[ass]
        x = points[idx]
        obj_list.append(tf.reduce_sum(tf.square(tf.subtract(m, x))))
    return tf.reduce_sum(obj_list)

obj = calculate_objective(centroids, assignments, data_points)


# Initialize
init = tf.global_variables_initializer()

j_values = []
j_values.append(0)
iter = 1
with tf.compat.v1.Session() as sess:
    sess.run(init)
    while (len(j_values) < 2 or abs(j_values[-1] - j_values[-2]) > EPSILON):
    #for step in range(20):
        distances_value, update_value, assignments_value, centroids_value, obj_value = sess.run(
            [
                distances,
                update,
                assignments,
                centroids,
                obj
            ]
        )
        group_count = []
        for ix in range(num_clusters):
            group_count.append(np.sum(assignments_value == ix))
        j_values.append(obj_value)
        axis_x.append(iter)
        iter += 1
        axis_y.append(obj_value)
    [centers, assignments] = sess.run([centroids, cluster_labels])

ax = plt.gca()

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
line = plt.plot(axis_x, axis_y, 'o-', label='Objective Function value')

plt.title('Objective Function value  / iteration')
plt.legend(loc='upper left',fontsize=11)
ax.set_xlabel('Iteration',fontsize=11,labelpad = 12.5)
ax.set_ylabel('Objective Function value',fontsize=11,labelpad = 12.5)

plt.grid(ax)
plt.savefig(fname='knn-iteration.png')

plt.show()

