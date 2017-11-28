"""
Tensorflow solution for the LOL win&lose prediction

Note:
Part of these code is inspired by MorvanZhou (https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/502_batch_normalization.py)
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf

ACTIVATION = tf.tanh
N_HIDDEN = 4
N_PER_LAYER = 20
learning_rate = 0.03
batch_norm_momentum = 0.99
training_steps = 500
B_INIT = tf.constant_initializer(-0.2)

# Load dataset
def load_data():
    dataset = pd.read_csv('../dataset/LOL/stats1.csv',index_col=False,
                          usecols=[
                              'win', 'kills','deaths','assists','longesttimespentliving','totdmgdealt',
                              'totheal','totdmgtaken','goldearned','totcctimedealt','champlvl'
                          ])
    return dataset

# Seperate the dataset into training set, cross validation set, and test set
def preprocessing(dataset):
    dataset.loc[:, 'kills'] /= 10
    dataset.loc[:, 'deaths'] /= 10
    dataset.loc[:, 'assists'] /= 10
    dataset.loc[:, 'longesttimespentliving'] /= 1000
    dataset.loc[:, 'totdmgdealt'] /= 100000
    dataset.loc[:, 'totheal'] /= 10000
    dataset.loc[:, 'totdmgtaken'] /= 10000
    dataset.loc[:, 'goldearned'] /= 10000
    dataset.loc[:, 'totcctimedealt'] /= 1000
    dataset.loc[:, 'champlvl'] /= 10
    y = dataset['win'].as_matrix()
    X = dataset.drop('win', axis=1).as_matrix()
    sss_test = StratifiedShuffleSplit(n_splits=3, test_size=0.2,random_state=0)
    sss_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.25,random_state=0)
    for train_cv_index, test_index in sss_test.split(X,y):
        X_train_cv, X_test = X[train_cv_index], X[test_index]
        y_train_cv, y_test = y[train_cv_index], y[test_index]
    for train_index, cv_index in sss_cv.split(X_train_cv,y_train_cv):
        X_train, X_cv = X[train_index], X[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
    return X_train,X_cv,X_test,y_train,y_cv,y_test



# Feed-forward neural network
class NN(object):
    def __init__(self, dropout_rate=0.1, batch_normalization=True):
        self.is_bn = batch_normalization

        self.w_init = tf.random_normal_initializer(0., .1)  # weights initialization
        self.pre_activation = [tf_x]
        if self.is_bn:
            self.layer_input = [tf.layers.batch_normalization(tf_x, training=tf_is_training)]  # for input data
        else:
            self.layer_input = [tf_x]
        for i in range(N_HIDDEN):  # adding hidden layers
            self.layer_input.append(self.add_layer(self.layer_input[-1], N_PER_LAYER, dropout_rate=dropout_rate, ac=ACTIVATION))
        self.out = tf.layers.dense(self.layer_input[-1], 2, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=self.out)

        self.accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(self.out, axis=1),)[1]

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        # Here choose an optimizer
            self.train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def add_layer(self, x, out_size, dropout_rate, ac=None):
        x = tf.layers.dense(x, out_size, kernel_initializer=self.w_init, bias_initializer=B_INIT)
        # self.pre_activation.append(x)
        # the momentum plays important rule.
        if self.is_bn: x = tf.layers.batch_normalization(x, momentum=batch_norm_momentum, training=tf_is_training)    # when have BN

        out = x if ac is None else ac(x)

        # Add dropout for regulization
        out = tf.layers.dropout(out, rate = dropout_rate, training=tf_is_training)
        return out


if __name__ == "__main__":
    # Load dataset and split data into training set, cross validation set and test set
    dataset = load_data()
    X_train, X_cv, X_test, y_train, y_cv, y_test = preprocessing(dataset)

    # tensorflow placeholder
    tf_x = tf.placeholder(tf.float32, [None, 10])
    tf_y = tf.placeholder(tf.int32, [None, 1])
    tf_is_training = tf.placeholder(tf.bool, None)

    # Initial neural network
    # nets = [NN(batch_normalization=False), NN(batch_normalization=True)]

    nets = [NN(dropout_rate=0.02), NN(dropout_rate=0.05)]

    # Start a session
    with tf.Session() as sess:
        # Initialize both global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)


        for step in range(training_steps):
            _, acc_train1, _, acc_train2, = sess.run([
                nets[0].train, nets[0].accuracy,
                nets[1].train, nets[1].accuracy
            ], feed_dict={tf_x: X_train, tf_y: y_train[:,np.newaxis], tf_is_training: True})

        acc_cv1, acc_cv2 = sess.run(
            [nets[0].accuracy,nets[1].accuracy],
            feed_dict={tf_x: X_cv, tf_y: y_cv[:,np.newaxis], tf_is_training: False}
        )

        # print("precision without batch_normalization:")
        # print("training: {}, cv: {}".format(acc_train_noBN, acc_cv_noBN))
        # print("precision with batch_normalization:")
        # print("training: {}, cv: {}".format(acc_train_BN, acc_cv_BN))

        print("train1: {}, cv1: {}".format(acc_train1,acc_cv1))
        print("train2: {}, cv2: {}".format(acc_train2, acc_cv2))
