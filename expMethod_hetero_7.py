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
    _follow = np.array(0)
    _co_loc = np.array(0)
    _co_link = np.array(0)
    _follow_follow = np.array(0)
    _follow_in = np.array(0)
    _follow_out = np.array(0)
    _followed_followed = np.array(0)
    _penalty_follow = np.array(0)
    _penalty_co_loc = np.array(0)
    _penalty_co_link = np.array(0)
    _penalty_follow_follow = np.array(0)
    _penalty_follow_in = np.array(0)
    _penalty_follow_out = np.array(0)
    _penalty_followed_followed = np.array(0)
    _num_examples = 0
    # epoch related variables
    _index_in_epoch = 0
    _epochs_completed = 0

    def __init__(self):
        print "init method"
        #self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

    def generate_batch(self, d1, d2, d3, d4, d5, d6, d7, n, batch_size):
	batch_index = random.sample(xrange(n), batch_size)
        return d1[batch_index], d2[batch_index], d3[batch_index], d4[batch_index], d5[batch_index], d6[batch_index], d7[batch_index]

    def set_penalty(self, data):
        beta = 2
        n_input = data.shape[1]
        # impose more penalty on errors of non-zero elements
        penalty = np.zeros((self._num_examples, n_input), dtype=float)
        for i in range(self._num_examples):
            for j in range(n_input):
                if (abs(data[i][j] - 0) > 0.0000000001):
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
            self._follow = self._follow[perm0]
            self._co_loc = self._co_loc[perm0]
            self._co_link = self._co_link[perm0]
            self._follow_follow = self._follow_follow[perm0]
            self._follow_in = self._follow_in[perm0]
            self._follow_out = self._follow_out[perm0]
            self._followed_followed = self._followed_followed[perm0]
            self._penalty_follow = self._penalty_follow[perm0]
            self._penalty_co_loc = self._penalty_co_loc[perm0]
            self._penalty_co_link = self._penalty_co_link[perm0]
            self._penalty_follow_follow = self._penalty_follow_follow[perm0]
            self._penalty_follow_in = self._penalty_follow_in[perm0]
            self._penalty_follow_out = self._penalty_follow_out[perm0]
            self._penalty_followed_followed = self._penalty_followed_followed[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            follow_rest_part = self._follow[start:self._num_examples]
            co_loc_rest_part = self._co_loc[start:self._num_examples]
            co_link_rest_part = self._co_link[start:self._num_examples]
            follow_follow_rest_part = self._follow_follow[start:self._num_examples]
            follow_in_rest_part = self._follow_in[start:self._num_examples]
            follow_out_rest_part = self._follow_out[start:self._num_examples]
            followed_followed_rest_part = self._followed_followed[start:self._num_examples]

            follow_penalty_rest_part = self._penalty_follow[start:self._num_examples]
            co_loc_penalty_rest_part = self._penalty_co_loc[start:self._num_examples]
            co_link_penalty_rest_part = self._penalty_co_link[start:self._num_examples]
            follow_follow_penalty_rest_part = self._penalty_follow_follow[start:self._num_examples]
            follow_in_penalty_rest_part = self._penalty_follow_in[start:self._num_examples]
            follow_out_penalty_rest_part = self._penalty_follow_out[start:self._num_examples]
            followed_followed_penalty_rest_part = self._penalty_followed_followed[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._follow = self._follow[perm]
                self._co_loc = self._co_loc[perm]
                self._co_link = self._co_link[perm]
                self._follow_follow = self._follow_follow[perm]
                self._follow_in = self._follow_in[perm]
                self._follow_out = self._follow_out[perm]
                self._followed_followed = self._followed_followed[perm]

                self._penalty_follow = self._penalty_follow[perm]
                self._penalty_co_loc = self._penalty_co_loc[perm]
                self._penalty_co_link = self._penalty_co_link[perm]
                self._penalty_follow_follow = self._penalty_follow_follow[perm]
                self._penalty_follow_in = self._penalty_follow_in[perm]
                self._penalty_follow_out = self._penalty_follow_out[perm]
                self._penalty_followed_followed = self._penalty_followed_followed[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            follow_new_part = self._follow[start:end]
            co_loc_new_part = self._co_loc[start:end]
            co_link_new_part = self._co_link[start:end]
            follow_follow_new_part = self._follow_follow[start:end]
            follow_in_new_part = self._follow_in[start:end]
            follow_out_new_part = self._follow_out[start:end]
            followed_followed_new_part = self._followed_followed[start:end]

            follow_penalty_new_part = self._penalty_follow[start:end]
            co_loc_penalty_new_part = self._penalty_co_loc[start:end]
            co_link_penalty_new_part = self._penalty_co_link[start:end]
            follow_follow_penalty_new_part = self._penalty_follow_follow[start:end]
            follow_in_penalty_new_part = self._penalty_follow_in[start:end]
            follow_out_penalty_new_part = self._penalty_follow_out[start:end]
            followed_followed_penalty_new_part = self._penalty_followed_followed[start:end]

            return np.concatenate((follow_rest_part, follow_new_part), axis=0), np.concatenate((co_loc_rest_part, co_loc_new_part), axis=0), np.concatenate((co_link_rest_part, co_link_new_part), axis=0), np.concatenate((follow_follow_rest_part, follow_follow_new_part), axis=0), np.concatenate((follow_in_rest_part, follow_in_new_part), axis=0), np.concatenate((follow_out_rest_part, follow_out_new_part), axis=0), np.concatenate((followed_followed_rest_part, followed_followed_new_part), axis=0) , np.concatenate((follow_penalty_rest_part, follow_penalty_new_part), axis=0), np.concatenate((co_loc_penalty_rest_part, co_loc_penalty_new_part), axis=0), np.concatenate((co_link_penalty_rest_part, co_link_penalty_new_part), axis=0), np.concatenate((follow_follow_penalty_rest_part, follow_follow_penalty_new_part), axis=0), np.concatenate((follow_in_penalty_rest_part, follow_in_penalty_new_part), axis=0), np.concatenate((follow_out_penalty_rest_part, follow_out_penalty_new_part), axis=0), np.concatenate((followed_followed_penalty_rest_part, followed_followed_penalty_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._follow[start:end], self._co_loc[start:end], self._co_link[start:end], self._follow_follow[start:end], self._follow_in[start:end], self._follow_out[start:end], self._followed_followed[start:end], self._penalty_follow[start:end], self._penalty_co_loc[start:end], self._penalty_co_link[start:end], self._penalty_follow_follow[start:end], self._penalty_follow_in[start:end], self._penalty_follow_out[start:end], self._penalty_followed_followed[start:end]

    def run(self, dataset):
        print "run foursquare heterogeneous method "
        sess = tf.Session(config = self.config)

        data_follow = dataset['foursquare']['follow_link']
        data_co_loc = dataset['foursquare']['co_list_loc']
        data_co_link = dataset['foursquare']['co_tip_loc']
        data_follow_follow = dataset['foursquare']['follow_follow']
        data_follow_in = dataset['foursquare']['follow_in']
        data_follow_out = dataset['foursquare']['follow_out']
        data_followed_followed = dataset['foursquare']['followed_followed']

        self._follow = data_follow
        self._co_loc = data_co_loc
        self._co_link = data_co_link
        self._follow_follow = data_follow_follow
        self._follow_in = data_follow_in
        self._follow_out = data_follow_out
        self._followed_followed = data_followed_followed

        self._num_examples = data_follow.shape[0]
        n_input = data_follow.shape[1]
        n_hidden_1 = 50
        n_hidden_2 = 50

        # Parameters
        learning_rate = 0.01
        training_epochs = 200
        batch_size = 256
        display_step = 1
        examples_to_show = 10

        # set penalties
        self._penalty_follow = self.set_penalty(data_follow)
        self._penalty_co_loc = self.set_penalty(data_co_loc)
        self._penalty_co_link = self.set_penalty(data_co_link)
        self._penalty_follow_follow = self.set_penalty(data_follow_follow)
        self._penalty_follow_in = self.set_penalty(data_follow_in)
        self._penalty_follow_out = self.set_penalty(data_follow_out)
        self._penalty_followed_followed = self.set_penalty(data_followed_followed)

        # input and variables
        X_follow = tf.placeholder("float", [None, n_input])
        X_co_loc = tf.placeholder("float", [None, n_input])
        X_co_link = tf.placeholder("float", [None, n_input])
        X_follow_follow = tf.placeholder("float", [None, n_input])
        X_follow_in = tf.placeholder("float", [None, n_input])
        X_follow_out = tf.placeholder("float", [None, n_input])
        X_followed_followed = tf.placeholder("float", [None, n_input])

        B_follow = tf.placeholder("float", [None, n_input])
        B_co_loc = tf.placeholder("float", [None, n_input])
        B_co_link = tf.placeholder("float", [None, n_input])
        B_follow_follow = tf.placeholder("float", [None, n_input])
        B_follow_in = tf.placeholder("float", [None, n_input])
        B_follow_out = tf.placeholder("float", [None, n_input])
        B_followed_followed = tf.placeholder("float", [None, n_input])

#        weights_follow = {
#            'encoder_h1': tf.get_variable('we1_f', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_f', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_f', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_f', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_co_loc = {
#            'encoder_h1': tf.get_variable('we1_cl', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_cl', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_cl', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_cl', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_co_link = {
#            'encoder_h1': tf.get_variable('we1_cli', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_cli', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_cli', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_cli', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_follow_follow = {
#            'encoder_h1': tf.get_variable('we1_ff', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_ff', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_ff', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_ff', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_follow_in = {
#            'encoder_h1': tf.get_variable('we1_fi', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_fi', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_fi', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_fi', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_follow_out = {
#            'encoder_h1': tf.get_variable('we1_fo', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_fo', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_fo', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_fo', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_followed_followed = {
#            'encoder_h1': tf.get_variable('we1_fd', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_h2': tf.get_variable('we2_fd', shape=[n_hidden_1, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd2_fd', shape=[n_hidden_2, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd3_df', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_concat = {
#            'encoder_h3': tf.get_variable('we2', shape=[n_hidden_2*7, n_hidden_3],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h3': tf.get_variable('wd2', shape=[n_hidden_3, n_hidden_2*7],
#                initializer = tf.contrib.layers.xavier_initializer()),
#        }

#        biases = {
#            'encoder_b1': tf.get_variable('be1', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'encoder_b2': tf.get_variable('be2', shape=[n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd2', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b2': tf.get_variable('bd3', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }

        weights_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_co_loc = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_co_link = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_follow_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_follow_in = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_follow_out = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_followed_followed = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_concat = {
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1*7, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1*7])),
        }
        biases_follow = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_co_loc = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_co_link = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_follow_follow = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_follow_in = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_follow_out = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_followed_followed = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_concat = {
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1*7])),
        }

        # Encoder Hidden layer with relu activation #1
        layer_1_follow = tf.sigmoid(tf.add(tf.matmul(X_follow,
                weights_follow['encoder_h1']), biases_follow['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_co_loc = tf.sigmoid(tf.add(tf.matmul(X_co_loc,
                weights_co_loc['encoder_h1']), biases_co_loc['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_co_link = tf.sigmoid(tf.add(tf.matmul(X_co_link,
                weights_co_link['encoder_h1']), biases_co_link['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_follow_follow = tf.sigmoid(tf.add(tf.matmul(X_follow_follow,
                weights_follow_follow['encoder_h1']), biases_follow_follow['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_follow_in = tf.sigmoid(tf.add(tf.matmul(X_follow_in,
                weights_follow_in['encoder_h1']), biases_follow_in['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_follow_out = tf.sigmoid(tf.add(tf.matmul(X_follow_out,
                weights_follow_out['encoder_h1']), biases_follow_out['encoder_b1']))

        # Encoder Hidden layer with relu activation #1
        layer_1_followed_followed = tf.sigmoid(tf.add(tf.matmul(X_followed_followed,
                weights_followed_followed['encoder_h1']), biases_followed_followed['encoder_b1']))

        # concat layer 1
        layer_1_concat = tf.concat([layer_1_follow, layer_1_co_loc, layer_1_co_link,
                layer_1_follow_follow, layer_1_follow_in, layer_1_follow_out, layer_1_followed_followed], 1)
        # concat encoder hidden layer with relu activation # 2
        layer_2_encoder = tf.sigmoid(tf.add(tf.matmul(layer_1_concat,
                weights_concat['encoder_h2']), biases_concat['encoder_b2']))
        # concat encoder hidden layer with relu activation # 2
        layer_2_decoder = tf.sigmoid(tf.add(tf.matmul(layer_2_encoder,
                weights_concat['decoder_h2']), biases_concat['decoder_b2']))

        # split layer 2 encoder result
        layer_2_decoder_follow = layer_2_decoder[:,0:n_hidden_2]
        layer_2_decoder_co_loc = layer_2_decoder[:,n_hidden_2:n_hidden_2*2]
        layer_2_decoder_co_link = layer_2_decoder[:,n_hidden_2*2:n_hidden_2*3]
        layer_2_decoder_follow_follow = layer_2_decoder[:,n_hidden_2*3:n_hidden_2*4]
        layer_2_decoder_follow_in = layer_2_decoder[:,n_hidden_2*4:n_hidden_2*5]
        layer_2_decoder_follow_out = layer_2_decoder[:,n_hidden_2*5:n_hidden_2*6]
        layer_2_decoder_followed_followed = layer_2_decoder[:,n_hidden_2*6:n_hidden_2*7]

        # Decoder Hidden layer with relu activation #3
        layer_3_follow = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_follow,
                weights_follow['decoder_h1']), biases_follow['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_co_loc = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_co_loc,
                weights_co_loc['decoder_h1']), biases_co_loc['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_co_link = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_co_link,
                weights_co_link['decoder_h1']), biases_co_link['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_follow_follow = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_follow_follow,
                weights_follow_follow['decoder_h1']), biases_follow_follow['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_follow_in = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_follow_in,
                weights_follow_in['decoder_h1']), biases_follow_in['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_follow_out = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_follow_out,
                weights_follow_out['decoder_h1']), biases_follow_out['decoder_b1']))

        # Decoder Hidden layer with relu activation #3
        layer_3_followed_followed = tf.sigmoid(tf.add(tf.matmul(layer_2_decoder_followed_followed,
                weights_followed_followed['decoder_h1']), biases_followed_followed['decoder_b1']))

	y_pred_follow = layer_3_follow
	y_true_follow = X_follow

	y_pred_co_loc = layer_3_co_loc
	y_true_co_loc = X_co_loc

	y_pred_co_link = layer_3_co_link
	y_true_co_link = X_co_link

	y_pred_follow_follow = layer_3_follow_follow
	y_true_follow_follow = X_follow_follow

	y_pred_follow_in = layer_3_follow_in
	y_true_follow_in = X_follow_in

	y_pred_follow_out = layer_3_follow_out
	y_true_follow_out = X_follow_out

	y_pred_followed_followed = layer_3_followed_followed
	y_true_followed_followed = X_followed_followed

        losses = [tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_follow , y_pred_follow) , B_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_co_loc , y_pred_co_loc) , B_co_loc)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_co_link , y_pred_co_link) , B_co_link)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_follow_follow , y_pred_follow_follow) , B_follow_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_follow_in , y_pred_follow_in) , B_follow_in)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_follow_out , y_pred_follow_out) , B_follow_out)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(y_true_followed_followed , y_pred_followed_followed) , B_followed_followed))
                 ]

        cost = tf.add_n(losses)
	optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

    	sess.run(init)

        pre_echos_completed = 0
        while(self._epochs_completed < training_epochs):
            batch_x_follow, batch_x_co_loc, batch_x_co_link, batch_x_follow_follow, batch_x_follow_in, batch_x_follow_out, batch_x_followed_followed, batch_penalty_follow, batch_penalty_co_loc, batch_penalty_co_link, batch_penalty_follow_follow, batch_penalty_follow_in, batch_penalty_follow_out, batch_penalty_followed_followed = self.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X_follow: batch_x_follow, X_co_loc: batch_x_co_loc, X_co_link: batch_x_co_link, X_follow_follow: batch_x_follow_follow, X_follow_in: batch_x_follow_in, X_follow_out: batch_x_follow_out, X_followed_followed: batch_x_followed_followed, B_follow: batch_penalty_follow, B_co_loc: batch_penalty_co_loc, B_co_link: batch_penalty_co_link, B_follow_follow: batch_penalty_follow_follow, B_follow_in: batch_penalty_follow_in, B_follow_out: batch_penalty_follow_out, B_followed_followed: batch_penalty_followed_followed})
            #x = sess.run(X_follow, feed_dict={X_follow: batch_x_follow})
            #p = sess.run(B_follow, feed_dict={B_follow: batch_penalty_follow})
            #print ' '.join(str(t) for t in x[1])
            #print ' '.join(str(t) for t in p[1])
            #for i in range(10):
            #    for j in range(len(x[0])):
            #        if x[i][j] != 0:
            #            print str(i) + '\t' + str(j) + '\t' + str(x[i][j]) + '\t' + str(p[i][j])
            #break
            ''' Display logs per epoch step'''
            if self._epochs_completed > pre_echos_completed:
                pre_echos_completed += 1
                print("Epoch:", '%04d' % (self._epochs_completed), "cost=", "{:.9f}".format(c))

#   	total_batch = int(n_input/batch_size)
#    	# Training cycle
#    	for epoch in range(training_epochs):
#            # Loop over all batches
#            for i in range(total_batch):
#		batch_x_follow, batch_x_co_loc, batch_x_co_link, batch_x_follow_follow, batch_x_follow_in, batch_x_follow_out, batch_x_followed_followed = self.generate_batch(data_follow, data_co_loc, data_co_link, data_follow_follow, data_follow_in, data_follow_out, data_followed_followed, n_input, batch_size)
#                # Run optimization op (backprop) and cost op (to get loss value)
#                _, c = sess.run([optimizer, cost], feed_dict={X_follow: batch_x_follow, X_co_loc: batch_x_co_loc, X_co_link: batch_x_co_link, X_follow_follow: batch_x_follow_follow, X_follow_in: batch_x_follow_in, X_follow_out: batch_x_follow_out, X_followed_followed: batch_x_followed_followed})
#            # Display logs per epoch step
#            if epoch % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        fp = open('embedding_feature','w')
        feature = sess.run(layer_2_encoder, feed_dict={X_follow: data_follow, X_co_loc: data_co_loc, X_co_link: data_co_link, X_follow_follow: data_follow_follow, X_follow_in: data_follow_in, X_follow_out: data_follow_out, X_followed_followed: data_followed_followed})
        for i in range(len(feature)):
            fp.write(' '.join(str(x) for x in feature[i])+'\n')
        fp.close()

        sess.close()
        return feature
