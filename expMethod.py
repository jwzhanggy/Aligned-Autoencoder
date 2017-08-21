from method import method
import numpy as np
import copy
from expResult import expResult
import pickle
from sklearn.cluster import KMeans
import os
import random
from scipy import linalg
import tensorflow as tf

class expMethod(method):

    # session configuration
    config = tf.ConfigProto()
    # data
    _follow_link = np.array(0)
    _penalty = np.array(0)
    _num_examples = 0
    # epoch related variables
    _index_in_epoch = 0
    _epochs_completed = 0

    def __init__(self):
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

    def set_penalty(self):
        beta = 2
        n_input = self._follow_link.shape[1]
        # impose more penalty on errors of non-zero elements
        penalty = np.zeros((self._num_examples, n_input), dtype=float)
        for i in range(self._num_examples):
            for j in range(n_input):
                if (abs(self._follow_link[i][j] - 0) > 0.0000000001):
                    penalty[i][j] = beta # beta > 1
                else:
                    penalty[i][j] = 1
        return penalty

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._follow_links = self._follow_link[perm0]
            self._penalty = self._penalty[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            follow_link_rest_part = self._follow_link[start:self._num_examples]
            penalty_rest_part = self._penalty[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._follow_links = self._follow_link[perm]
                self._penalty = self._penalty[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            follow_link_new_part = self._follow_link[start:end]
            penalty_new_part = self._penalty[start:end]
            return np.concatenate((follow_link_rest_part, follow_link_new_part), axis=0), np.concatenate((penalty_rest_part, penalty_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._follow_link[start:end], self._penalty[start:end]

    def run(self, dataset):
        print "run foursquare follow method "
        sess = tf.Session(config = self.config)

        data = dataset['foursquare']['follow_link']
        self._follow_link = data

        self._num_examples = data.shape[0]
        self._penalty = self.set_penalty()
        print self._penalty.shape

        # Parameters
        n_input = data.shape[1]
        n_hidden_1 = 50
        learning_rate = 0.001
        training_epochs = 50
        batch_size = 128
        display_step = 1
        examples_to_show = 10

        # input and variables
        X = tf.placeholder("float", [None, n_input])
        B = tf.placeholder("float", [None, n_input])
#        weights = {
#            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
#        }
#        biases = {
#            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
#            'decoder_b1': tf.Variable(tf.zeros([n_input])),
#        }
        weights = {
            'encoder_h1': tf.get_variable('we1', shape=[n_input, n_hidden_1],
                initializer = tf.contrib.layers.xavier_initializer()),
            'decoder_h1': tf.get_variable('wd1', shape=[n_hidden_1, n_input],
                initializer = tf.contrib.layers.xavier_initializer())
        }
        biases = {
            'encoder_b1': tf.get_variable('be1', shape=[n_hidden_1],
                initializer = tf.contrib.layers.xavier_initializer()),
            'decoder_b1': tf.get_variable('bd1', shape=[n_input],
                initializer = tf.contrib.layers.xavier_initializer())
        }

        # Encoder Hidden layer with relu activation #1
        layer_1 = tf.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']),
                                       biases['encoder_b1']))

        # Decoder Hidden layer with relu activation #4
        layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h1']),
                                       biases['decoder_b1']))

        y_pred = layer_2
        y_true = X

        #cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        #cost = tf.reduce_mean(tf.div(tf.reduce_sum(tf.pow(y_true - y_pred, 2)), 2))
        #cost = tf.nn.l2_loss(y_true - y_pred)

        # only penatly error on non-zero elements
        #cost = tf.nn.l2_loss(tf.multiply(tf.subtract(y_true , y_pred) , y_true))

        # add penalty error on non-zero elements
        cost = tf.nn.l2_loss(tf.multiply(tf.subtract(y_true , y_pred) , B))

        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        sess.run(init)

        pre_echos_completed = 0
        while(self._epochs_completed < training_epochs):
            batch_xs, batch_penalty = self.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, B: batch_penalty})
            ''' Display logs per epoch step'''
            if self._epochs_completed > pre_echos_completed:
                pre_echos_completed += 1
                print("Epoch:", '%04d' % (self._epochs_completed), "cost=", "{:.9f}".format(c))

#    # Training cycle
#       total_batch = int(n_input/batch_size)
#        for epoch in range(training_epochs):
#            # Loop over all batches
#            for i in range(total_batch):
#        batch_xs = self.next_batch(data, n_input, i, batch_size)
#                # Run optimization op (backprop) and cost op (to get loss value)
#                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
#                feature = sess.run(layer_2, feed_dict={X: batch_xs})
#            # Display logs per epoch step
#            if epoch % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

        feature = sess.run(layer_1, feed_dict={X: data})

        fp = open('embedding_feature','w')
        for i in range(len(feature)):
            fp.write(' '.join(str(x) for x in feature[i])+'\n')
        fp.close()

        sess.close()
        return feature
