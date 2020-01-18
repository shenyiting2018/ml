import matplotlib
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


tf.disable_v2_behavior()


FILE_PATH = './pima-indians-diabetes.csv'
NUM_EXP = 1000
axis_x = []
axis_y = []


def split_data(dataset):
    num_attributes = dataset.shape[1] - 1 # last column is label
    index_label = num_attributes # last column is label

    # Use Python list comprehension
    data_label_0 = [line for line in dataset if line[index_label] == 0]
    data_label_1 = [line for line in dataset if line[index_label] == 1]

    # Do a quick assertion test
    for data in data_label_0:
        assert data[index_label] == 0
    for data in data_label_1:
        assert data[index_label] == 1

    return data_label_0, data_label_1


def generate_random_sample_index(sample_size, dataset_size):
    sample = np.random.choice(
        dataset_size,
        size=sample_size,
        replace=False,
    )
    return sample


def generate_training_and_test_dataset(data_label_0, data_label_1, sample_size):
    training_index_label_0 = generate_random_sample_index(sample_size, len(data_label_0))
    training_index_label_1 = generate_random_sample_index(sample_size, len(data_label_1))

    # Make sure shape of indices is correct
    assert len(training_index_label_0) == sample_size
    assert len(training_index_label_1) == sample_size

    test_index_label_0 = list(set(range(len(data_label_0))) - set(training_index_label_0))
    test_index_label_1 = list(set(range(len(data_label_1))) - set(training_index_label_1))

    assert len(test_index_label_0) == len(data_label_0) - sample_size
    assert len(test_index_label_1) == len(data_label_1) - sample_size

    # ref:https://stackoverflow.com/questions/19821425/how-to-filter-numpy-array-by-list-of-indices
    training_data_label_0 = np.array(data_label_0)[training_index_label_0]
    training_data_label_1 = np.array(data_label_1)[training_index_label_1]
    test_data_label_0 = np.array(data_label_0)[test_index_label_0]
    test_data_label_1 = np.array(data_label_1)[test_index_label_1]

    # merge two dataset
    training_data = np.vstack((training_data_label_0, training_data_label_1))
    test_data = np.vstack((test_data_label_0, test_data_label_1))

    assert len(training_data) == len(training_data_label_0) + len(training_data_label_1)
    assert len(test_data) == len(test_data_label_0) + len(test_data_label_1)

    # import ipdb; ipdb.set_trace()
    training_attr = np.array([single_example[:-1] for single_example in training_data])
    training_label = np.array([[single_example[-1]] for single_example in training_data])
    test_attr = np.array([single_example[:-1] for single_example in test_data])
    test_label = np.array([[single_example[-1]] for single_example in test_data])

    return training_attr, training_label, test_attr, test_label


"""Beginning of excutable script
"""
dataset_without_bias = pd.read_csv(FILE_PATH, header=None)
num_examples = dataset_without_bias.shape[0]
dataset = np.c_[np.ones((num_examples, 1)), dataset_without_bias]
# The two dataset without different labels
data_label_0, data_label_1 = split_data(dataset)


for sample_size in range(40, 201, 40):
    accuracy_rate = []
    for exp in range(NUM_EXP):
        training_attr, training_label, test_attr, test_label = generate_training_and_test_dataset(
            data_label_0,
            data_label_1,
            sample_size=sample_size,
        )
        n_train, m = training_attr.shape
        n_test, m = test_attr.shape

        X = tf.placeholder(tf.float64, shape = (None, m), name = 'X')
        t = tf.placeholder(tf.float64, shape = (None, 1), name = 't')
        n = tf.placeholder(tf.float64, name = 'n')
        XT = tf.transpose(X)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)

        y = tf.matmul(X, w)


        MSE = tf.div(tf.matmul(tf.transpose(y - t), y - t), n)

        w_star = tf.placeholder(tf.float64, shape=(m, 1), name="w_star")
        y_test = tf.matmul(X, w_star)
        y_test_predicted = tf.round(y_test)


        #MSE_test = tf.abs(y - t)
        MSE_test = tf.abs(y_test_predicted - t)
        #MSE_test = tf.div(tf.matmul(tf.transpose(y_test - t), y_test - t), n)


        with tf.Session() as sess:
            MSE_train_val, w_val = sess.run(
                [MSE, w],
                feed_dict={
                    X: training_attr,
                    t: training_label,
                    n: n_train,
                },
            )

            MSE_test_val, y_test_val, y_test_predicted_val = sess.run(
                [MSE_test, y_test, y_test_predicted],
                feed_dict={
                    X: test_attr,
                    t: test_label,
                    n: n_test,
                    w_star: w_val,
                },
            )

        error_count = np.sum(MSE_test_val)
        accuracy_rate.append(1 - (float(error_count) / n_test))

    print(
        "n = {}, error rate = {}".format(
            sample_size,
            np.mean(np.array(accuracy_rate)),
        )
    )
    axis_x.append(sample_size)
    axis_y.append(np.mean(np.array(accuracy_rate)))

ax = plt.gca()

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
line = plt.plot(axis_x, axis_y, 'o-', label='Accuracy')

plt.title('Accuracy rate / sample size')
plt.legend(loc='lower right',fontsize=11)
ax.set_xlabel('Values of n',fontsize=11,labelpad = 12.5)
ax.set_ylabel('Accuracy',fontsize=11,labelpad = 12.5)

plt.grid(ax)
plt.savefig(fname='accuracy-rate.png')

plt.show()


