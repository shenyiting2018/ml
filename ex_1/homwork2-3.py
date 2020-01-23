import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd


tf.disable_v2_behavior()

FILE_PATH = './Iris.csv'

dataset_initial = pd.read_csv(FILE_PATH)
num_sample = dataset_initial.shape[0]
num_attr = dataset_initial.shape[1] - 2
num_cluster = 3

df = pd.DataFrame(dataset_initial)
dataset = np.array(df.drop(['Sample Index', 'outcome(Cluster Index)'], axis=1))



def create_samples():
    points = tf.Variable(initial_value=dataset, name='points')
    #centroids = tf.Variable(tf.slice(tf.random.shuffle(points), [0, 0], [num_cluster, num_attr]))
    return points


def choose_random_centroids(samples, n_clusters):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples))
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids


def update_centroids(samples, nearest_indices, n_clusters, distances):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids, distances


def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample

    # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
               tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    nearest_indices = mins
    distances = tf.reduce_sum(distances)
    return nearest_indices, distances


samples = create_samples()
initial_centroids = choose_random_centroids(samples, num_cluster)
nearest_indices, distances = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, num_cluster, distances)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    initial_centroids_value = session.run(initial_centroids)
    samples_value = session.run(samples)

    for step in range(10):
        #print("samples_value", samples_value)
        print("initial_centroids_value", initial_centroids_value)

        updated_centroid_value, distances_value = session.run(updated_centroids)
        print("updated_centroid_value", updated_centroid_value)
        print("distances_value", distances_value)

# plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)