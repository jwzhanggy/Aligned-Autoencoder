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
    dataset_prefix = '../data/link_data'

    # data
    _twitter_follow = np.array(0)
    _twitter_co_loc = np.array(0)
    _twitter_co_link = np.array(0)
    _twitter_follow_follow = np.array(0)
    _twitter_follow_in = np.array(0)
    _twitter_follow_out = np.array(0)
    _twitter_followed_followed = np.array(0)

    _foursquare_follow = np.array(0)
    _foursquare_co_loc = np.array(0)
    _foursquare_co_link = np.array(0)
    _foursquare_follow_follow = np.array(0)
    _foursquare_follow_in = np.array(0)
    _foursquare_follow_out = np.array(0)
    _foursquare_followed_followed = np.array(0)
    _foursquare_follow = np.array(0)

    _twitter_penalty_follow = np.array(0)
    _twitter_penalty_co_loc = np.array(0)
    _twitter_penalty_co_link = np.array(0)
    _twitter_penalty_follow_follow = np.array(0)
    _twitter_penalty_follow_in = np.array(0)
    _twitter_penalty_follow_out = np.array(0)
    _twitter_penalty_followed_followed = np.array(0)

    _penalty_foursquare_follow = np.array(0)
    _penalty_foursquare_co_loc = np.array(0)
    _penalty_foursquare_co_link = np.array(0)
    _penalty_foursquare_follow_follow = np.array(0)
    _penalty_foursquare_follow_in = np.array(0)
    _penalty_foursquare_follow_out = np.array(0)
    _penalty_foursquare_followed_followed = np.array(0)

    _num_examples = 0
    # epoch related variables
    _index_in_epoch = 0
    _epochs_completed = 0

    def __init__(self):
        print "init method"
        #self.config.log_device_placement = True
        #self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

    def generate_batch(self, n, batch_size):
	batch_index = random.sample(xrange(n), batch_size)
        return batch_index

    def set_penalty(self, data):
        beta = 100
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

            self._twitter_follow = self._twitter_follow[perm0]
            self._twitter_co_loc = self._twitter_co_loc[perm0]
            self._twitter_co_link = self._twitter_co_link[perm0]
            self._twitter_follow_follow = self._twitter_follow_follow[perm0]
            self._twitter_follow_in = self._twitter_follow_in[perm0]
            self._twitter_follow_out = self._twitter_follow_out[perm0]
            self._twitter_followed_followed = self._twitter_followed_followed[perm0]

            self._foursquare_follow = self._foursquare_follow[perm0]
            self._foursquare_co_loc = self._foursquare_co_loc[perm0]
            self._foursquare_co_link = self._foursquare_co_link[perm0]
            self._foursquare_follow_follow = self._foursquare_follow_follow[perm0]
            self._foursquare_follow_in = self._foursquare_follow_in[perm0]
            self._foursquare_follow_out = self._foursquare_follow_out[perm0]
            self._foursquare_followed_followed = self._foursquare_followed_followed[perm0]

            self._twitter_penalty_follow = self._twitter_penalty_follow[perm0]
            self._twitter_penalty_co_loc = self._twitter_penalty_co_loc[perm0]
            self._twitter_penalty_co_link = self._twitter_penalty_co_link[perm0]
            self._twitter_penalty_follow_follow = self._twitter_penalty_follow_follow[perm0]
            self._twitter_penalty_follow_in = self._twitter_penalty_follow_in[perm0]
            self._twitter_penalty_follow_out = self._twitter_penalty_follow_out[perm0]
            self._twitter_penalty_followed_followed = self._twitter_penalty_followed_followed[perm0]

            self._foursquare_penalty_follow = self._foursquare_penalty_follow[perm0]
            self._foursquare_penalty_co_loc = self._foursquare_penalty_co_loc[perm0]
            self._foursquare_penalty_co_link = self._foursquare_penalty_co_link[perm0]
            self._foursquare_penalty_follow_follow = self._foursquare_penalty_follow_follow[perm0]
            self._foursquare_penalty_follow_in = self._foursquare_penalty_follow_in[perm0]
            self._foursquare_penalty_follow_out = self._foursquare_penalty_follow_out[perm0]
            self._foursquare_penalty_followed_followed = self._foursquare_penalty_followed_followed[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            follow_twitter_rest_part = self._twitter_follow[start:self._num_examples]
            co_loc_twitter_rest_part = self._twitter_co_loc[start:self._num_examples]
            co_link_twitter_rest_part = self._twitter_co_link[start:self._num_examples]
            follow_follow_twitter_rest_part = self._twitter_follow_follow[start:self._num_examples]
            follow_in_twitter_rest_part = self._twitter_follow_in[start:self._num_examples]
            follow_out_twitter_rest_part = self._twitter_follow_out[start:self._num_examples]
            followed_followed_twitter_rest_part = self._twitter_followed_followed[start:self._num_examples]

            follow_foursquare_rest_part = self._foursquare_follow[start:self._num_examples]
            co_loc_foursquare_rest_part = self._foursquare_co_loc[start:self._num_examples]
            co_link_foursquare_rest_part = self._foursquare_co_link[start:self._num_examples]
            follow_follow_foursquare_rest_part = self._foursquare_follow_follow[start:self._num_examples]
            follow_in_foursquare_rest_part = self._foursquare_follow_in[start:self._num_examples]
            follow_out_foursquare_rest_part = self._foursquare_follow_out[start:self._num_examples]
            followed_followed_foursquare_rest_part = self._foursquare_followed_followed[start:self._num_examples]

            follow_twitter_penalty_rest_part = self._twitter_penalty_follow[start:self._num_examples]
            co_loc_twitter_penalty_rest_part = self._twitter_penalty_co_loc[start:self._num_examples]
            co_link_twitter_penalty_rest_part = self._twitter_penalty_co_link[start:self._num_examples]
            follow_follow_twitter_penalty_rest_part = self._twitter_penalty_follow_follow[start:self._num_examples]
            follow_in_twitter_penalty_rest_part = self._twitter_penalty_follow_in[start:self._num_examples]
            follow_out_twitter_penalty_rest_part = self._twitter_penalty_follow_out[start:self._num_examples]
            followed_followed_twitter_penalty_rest_part = self._twitter_penalty_followed_followed[start:self._num_examples]

            follow_foursquare_penalty_rest_part = self._foursquare_penalty_follow[start:self._num_examples]
            co_loc_foursquare_penalty_rest_part = self._foursquare_penalty_co_loc[start:self._num_examples]
            co_link_foursquare_penalty_rest_part = self._foursquare_penalty_co_link[start:self._num_examples]
            follow_follow_foursquare_penalty_rest_part = self._foursquare_penalty_follow_follow[start:self._num_examples]
            follow_in_foursquare_penalty_rest_part = self._foursquare_penalty_follow_in[start:self._num_examples]
            follow_out_foursquare_penalty_rest_part = self._foursquare_penalty_follow_out[start:self._num_examples]
            followed_followed_foursquare_penalty_rest_part = self._foursquare_penalty_followed_followed[start:self._num_examples]

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._twitter_follow = self._twitter_follow[perm]
                self._twitter_co_loc = self._twitter_co_loc[perm]
                self._twitter_co_link = self._twitter_co_link[perm]
                self._twitter_follow_follow = self._twitter_follow_follow[perm]
                self._twitter_follow_in = self._twitter_follow_in[perm]
                self._twitter_follow_out = self._twitter_follow_out[perm]
                self._twitter_followed_followed = self._twitter_followed_followed[perm]

                self._foursquare_follow = self._foursquare_follow[perm]
                self._foursquare_co_loc = self._foursquare_co_loc[perm]
                self._foursquare_co_link = self._foursquare_co_link[perm]
                self._foursquare_follow_follow = self._foursquare_follow_follow[perm]
                self._foursquare_follow_in = self._foursquare_follow_in[perm]
                self._foursquare_follow_out = self._foursquare_follow_out[perm]
                self._foursquare_followed_followed = self._foursquare_followed_followed[perm]

                self._twitter_penalty_follow = self._twitter_penalty_follow[perm]
                self._twitter_penalty_co_loc = self._twitter_penalty_co_loc[perm]
                self._twitter_penalty_co_link = self._twitter_penalty_co_link[perm]
                self._twitter_penalty_follow_follow = self._twitter_penalty_follow_follow[perm]
                self._twitter_penalty_follow_in = self._twitter_penalty_follow_in[perm]
                self._twitter_penalty_follow_out = self._twitter_penalty_follow_out[perm]
                self._twitter_penalty_followed_followed = self._twitter_penalty_followed_followed[perm]

                self._foursquare_penalty_follow = self._foursquare_penalty_follow[perm]
                self._foursquare_penalty_co_loc = self._foursquare_penalty_co_loc[perm]
                self._foursquare_penalty_co_link = self._foursquare_penalty_co_link[perm]
                self._foursquare_penalty_follow_follow = self._foursquare_penalty_follow_follow[perm]
                self._foursquare_penalty_follow_in = self._foursquare_penalty_follow_in[perm]
                self._foursquare_penalty_follow_out = self._foursquare_penalty_follow_out[perm]
                self._foursquare_penalty_followed_followed = self._foursquare_penalty_followed_followed[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            follow_twitter_new_part = self._twitter_follow[start:end]
            co_loc_twitter_new_part = self._twitter_co_loc[start:end]
            co_link_twitter_new_part = self._twitter_co_link[start:end]
            follow_follow_twitter_new_part = self._twitter_follow_follow[start:end]
            follow_in_twitter_new_part = self._twitter_follow_in[start:end]
            follow_out_twitter_new_part = self._twitter_follow_out[start:end]
            followed_followed_twitter_new_part = self._twitter_followed_followed[start:end]

            follow_foursquare_new_part = self._foursquare_follow[start:end]
            co_loc_foursquare_new_part = self._foursquare_co_loc[start:end]
            co_link_foursquare_new_part = self._foursquare_co_link[start:end]
            follow_follow_foursquare_new_part = self._foursquare_follow_follow[start:end]
            follow_in_foursquare_new_part = self._foursquare_follow_in[start:end]
            follow_out_foursquare_new_part = self._foursquare_follow_out[start:end]
            followed_followed_foursquare_new_part = self._foursquare_followed_followed[start:end]

            follow_twitter_penalty_new_part = self._twitter_penalty_follow[start:end]
            co_loc_twitter_penalty_new_part = self._twitter_penalty_co_loc[start:end]
            co_link_twitter_penalty_new_part = self._twitter_penalty_co_link[start:end]
            follow_follow_twitter_penalty_new_part = self._twitter_penalty_follow_follow[start:end]
            follow_in_twitter_penalty_new_part = self._twitter_penalty_follow_in[start:end]
            follow_out_twitter_penalty_new_part = self._twitter_penalty_follow_out[start:end]
            followed_followed_twitter_penalty_new_part = self._twitter_penalty_followed_followed[start:end]

            follow_foursquare_penalty_new_part = self._foursquare_penalty_follow[start:end]
            co_loc_foursquare_penalty_new_part = self._foursquare_penalty_co_loc[start:end]
            co_link_foursquare_penalty_new_part = self._foursquare_penalty_co_link[start:end]
            follow_follow_foursquare_penalty_new_part = self._foursquare_penalty_follow_follow[start:end]
            follow_in_foursquare_penalty_new_part = self._foursquare_penalty_follow_in[start:end]
            follow_out_foursquare_penalty_new_part = self._foursquare_penalty_follow_out[start:end]
            followed_followed_foursquare_penalty_new_part = self._foursquare_penalty_followed_followed[start:end]

            return np.concatenate((follow_twitter_rest_part, follow_twitter_new_part), axis=0), np.concatenate((co_loc_twitter_rest_part, co_loc_twitter_new_part), axis=0), np.concatenate((co_link_twitter_rest_part, co_link_twitter_new_part), axis=0), np.concatenate((follow_follow_twitter_rest_part, follow_follow_twitter_new_part), axis=0), np.concatenate((follow_in_twitter_rest_part, follow_in_twitter_new_part), axis=0), np.concatenate((follow_out_twitter_rest_part, follow_out_twitter_new_part), axis=0), np.concatenate((followed_followed_twitter_rest_part, followed_followed_twitter_new_part), axis=0), np.concatenate((follow_foursquare_rest_part, follow_foursquare_new_part), axis=0), np.concatenate((co_loc_foursquare_rest_part, co_loc_foursquare_new_part), axis=0), np.concatenate((co_link_foursquare_rest_part, co_link_foursquare_new_part), axis=0), np.concatenate((follow_follow_foursquare_rest_part, follow_follow_foursquare_new_part), axis=0), np.concatenate((follow_in_foursquare_rest_part, follow_in_foursquare_new_part), axis=0), np.concatenate((follow_out_foursquare_rest_part, follow_out_foursquare_new_part), axis=0), np.concatenate((followed_followed_foursquare_rest_part, followed_followed_foursquare_new_part), axis=0), np.concatenate((follow_twitter_penalty_rest_part, follow_twitter_penalty_new_part), axis=0), np.concatenate((co_loc_twitter_penalty_rest_part, co_loc_twitter_penalty_new_part), axis=0), np.concatenate((co_link_twitter_penalty_rest_part, co_link_twitter_penalty_new_part), axis=0), np.concatenate((follow_follow_twitter_penalty_rest_part, follow_follow_twitter_penalty_new_part), axis=0), np.concatenate((follow_in_twitter_penalty_rest_part, follow_in_twitter_penalty_new_part), axis=0), np.concatenate((follow_out_twitter_penalty_rest_part, follow_out_twitter_penalty_new_part), axis=0), np.concatenate((followed_followed_twitter_penalty_rest_part, followed_followed_twitter_penalty_new_part), axis=0), np.concatenate((follow_foursquare_penalty_rest_part, follow_foursquare_penalty_new_part), axis=0), np.concatenate((co_loc_foursquare_penalty_rest_part, co_loc_foursquare_penalty_new_part), axis=0), np.concatenate((co_link_foursquare_penalty_rest_part, co_link_foursquare_penalty_new_part), axis=0), np.concatenate((follow_follow_foursquare_penalty_rest_part, follow_follow_foursquare_penalty_new_part), axis=0), np.concatenate((follow_in_foursquare_penalty_rest_part, follow_in_foursquare_penalty_new_part), axis=0), np.concatenate((follow_out_foursquare_penalty_rest_part, follow_out_foursquare_penalty_new_part), axis=0), np.concatenate((followed_followed_foursquare_penalty_rest_part, followed_followed_foursquare_penalty_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._twitter_follow[start:end], self._twitter_co_loc[start:end], self._twitter_co_link[start:end], self._twitter_follow_follow[start:end], self._twitter_follow_in[start:end], self._twitter_follow_out[start:end], self._twitter_followed_followed[start:end], self._foursquare_follow[start:end], self._foursquare_co_loc[start:end], self._foursquare_co_link[start:end], self._foursquare_follow_follow[start:end], self._foursquare_follow_in[start:end], self._foursquare_follow_out[start:end], self._foursquare_followed_followed[start:end], self._twitter_penalty_follow[start:end], self._twitter_penalty_co_loc[start:end], self._twitter_penalty_co_link[start:end], self._twitter_penalty_follow_follow[start:end], self._twitter_penalty_follow_in[start:end], self._twitter_penalty_follow_out[start:end], self._twitter_penalty_followed_followed[start:end], self._foursquare_penalty_follow[start:end], self._foursquare_penalty_co_loc[start:end], self._foursquare_penalty_co_link[start:end], self._foursquare_penalty_follow_follow[start:end], self._foursquare_penalty_follow_in[start:end], self._foursquare_penalty_follow_out[start:end], self._foursquare_penalty_followed_followed[start:end]

    def run(self, dataset):
        print "run twitter heterogeneous method "
        sess = tf.Session(config = self.config)

        twitter_follow = dataset['twitter']['follow_link']
        twitter_co_loc = dataset['twitter']['co_loc']
        twitter_co_link = dataset['twitter']['contact_link']
        twitter_follow_follow = dataset['twitter']['follow_follow']
        twitter_follow_in = dataset['twitter']['follow_in']
        twitter_follow_out = dataset['twitter']['follow_out']
        twitter_followed_followed = dataset['twitter']['followed_followed']

        self._twitter_follow = twitter_follow
        self._twitter_co_loc = twitter_co_loc
        self._twitter_co_link = twitter_co_link
        self._twitter_follow_follow = twitter_follow_follow
        self._twitter_follow_in = twitter_follow_in
        self._twitter_follow_out = twitter_follow_out
        self._twitter_followed_followed = twitter_followed_followed

        foursquare_follow = dataset['foursquare']['follow_link']
        foursquare_co_loc = dataset['foursquare']['co_list_loc']
        foursquare_co_link = dataset['foursquare']['co_tip_loc']
        foursquare_follow_follow = dataset['foursquare']['follow_follow']
        foursquare_follow_in = dataset['foursquare']['follow_in']
        foursquare_follow_out = dataset['foursquare']['follow_out']
        foursquare_followed_followed = dataset['foursquare']['followed_followed']

        self._foursquare_follow = foursquare_follow
        self._foursquare_co_loc = foursquare_co_loc
        self._foursquare_co_link = foursquare_co_link
        self._foursquare_follow_follow = foursquare_follow_follow
        self._foursquare_follow_in = foursquare_follow_in
        self._foursquare_follow_out = foursquare_follow_out
        self._foursquare_followed_followed = foursquare_followed_followed

        self._num_examples = twitter_follow.shape[0]
        instance_num = twitter_follow.shape[0]
        n_input = twitter_follow.shape[1]
        n_hidden_1 = 50
        n_hidden_2 = 50

        # Parameters
        learning_rate = 0.001
        training_epochs = 600
        batch_size = 64
        display_step = 1

        # set penalties
        self._twitter_penalty_follow = self.set_penalty(twitter_follow)
        self._twitter_penalty_co_loc = self.set_penalty(twitter_co_loc)
        self._twitter_penalty_co_link = self.set_penalty(twitter_co_link)
        self._twitter_penalty_follow_follow = self.set_penalty(twitter_follow_follow)
        self._twitter_penalty_follow_in = self.set_penalty(twitter_follow_in)
        self._twitter_penalty_follow_out = self.set_penalty(twitter_follow_out)
        self._twitter_penalty_followed_followed = self.set_penalty(twitter_followed_followed)

        # set penalties
        self._foursquare_penalty_follow = self.set_penalty(foursquare_follow)
        self._foursquare_penalty_co_loc = self.set_penalty(foursquare_co_loc)
        self._foursquare_penalty_co_link = self.set_penalty(foursquare_co_link)
        self._foursquare_penalty_follow_follow = self.set_penalty(foursquare_follow_follow)
        self._foursquare_penalty_follow_in = self.set_penalty(foursquare_follow_in)
        self._foursquare_penalty_follow_out = self.set_penalty(foursquare_follow_out)
        self._foursquare_penalty_followed_followed = self.set_penalty(foursquare_followed_followed)

        # input and variables
        X1_follow = tf.placeholder("float", [None, n_input])
        X1_co_loc = tf.placeholder("float", [None, n_input])
        X1_co_link = tf.placeholder("float", [None, n_input])
        X1_follow_follow = tf.placeholder("float", [None, n_input])
        X1_follow_in = tf.placeholder("float", [None, n_input])
        X1_follow_out = tf.placeholder("float", [None, n_input])
        X1_followed_followed = tf.placeholder("float", [None, n_input])

        X2_follow = tf.placeholder("float", [None, n_input])
        X2_co_loc = tf.placeholder("float", [None, n_input])
        X2_co_link = tf.placeholder("float", [None, n_input])
        X2_follow_follow = tf.placeholder("float", [None, n_input])
        X2_follow_in = tf.placeholder("float", [None, n_input])
        X2_follow_out = tf.placeholder("float", [None, n_input])
        X2_followed_followed = tf.placeholder("float", [None, n_input])

        B1_follow = tf.placeholder("float", [None, n_input])
        B1_co_loc = tf.placeholder("float", [None, n_input])
        B1_co_link = tf.placeholder("float", [None, n_input])
        B1_follow_follow = tf.placeholder("float", [None, n_input])
        B1_follow_in = tf.placeholder("float", [None, n_input])
        B1_follow_out = tf.placeholder("float", [None, n_input])
        B1_followed_followed = tf.placeholder("float", [None, n_input])

        B2_follow = tf.placeholder("float", [None, n_input])
        B2_co_loc = tf.placeholder("float", [None, n_input])
        B2_co_link = tf.placeholder("float", [None, n_input])
        B2_follow_follow = tf.placeholder("float", [None, n_input])
        B2_follow_in = tf.placeholder("float", [None, n_input])
        B2_follow_out = tf.placeholder("float", [None, n_input])
        B2_followed_followed = tf.placeholder("float", [None, n_input])

#        weights_twitter_follow = {
#            'encoder_h1': tf.get_variable('we1_tf', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tf', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_co_loc = {
#            'encoder_h1': tf.get_variable('we1_tcl', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tcl', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_co_link = {
#            'encoder_h1': tf.get_variable('we1_tclk', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tclk', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_follow_follow = {
#            'encoder_h1': tf.get_variable('we1_tff', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tff', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_follow_in = {
#            'encoder_h1': tf.get_variable('we1_tfi', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tfi', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_follow_out = {
#            'encoder_h1': tf.get_variable('we1_tfo', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tfo', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_followed_followed = {
#            'encoder_h1': tf.get_variable('we1_tfed', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_tfed', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_twitter_concat = {
#            'encoder_h2': tf.get_variable('we1_tcc', shape=[n_hidden_1*7, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd1_tcc', shape=[n_hidden_2, n_hidden_1*7],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#
#        biases_twitter_follow = {
#            'encoder_b1': tf.get_variable('be1_tf', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tf', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_co_loc = {
#            'encoder_b1': tf.get_variable('be1_tcl', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tcl', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_co_link = {
#            'encoder_b1': tf.get_variable('be1_tclk', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tclk', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_follow_follow = {
#            'encoder_b1': tf.get_variable('be1_tff', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tff', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_follow_in = {
#            'encoder_b1': tf.get_variable('be1_tfi', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tfi', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_follow_out = {
#            'encoder_b1': tf.get_variable('be1_tfo', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tfo', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_followed_followed = {
#            'encoder_b1': tf.get_variable('be1_tfed', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_tfed', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_twitter_concat = {
#            'encoder_b2': tf.get_variable('be1_tcc', shape=[n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b2': tf.get_variable('bd1_tcc', shape=[n_hidden_1*7],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#
#
#        weights_foursquare_follow = {
#            'encoder_h1': tf.get_variable('we1_ff', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_ff', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_co_loc = {
#            'encoder_h1': tf.get_variable('we1_fcl', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_fcl', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_co_link = {
#            'encoder_h1': tf.get_variable('we1_fclk', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_fclk', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_follow_follow = {
#            'encoder_h1': tf.get_variable('we1_fff', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_fff', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_follow_in = {
#            'encoder_h1': tf.get_variable('we1_ffi', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_ffi', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_follow_out = {
#            'encoder_h1': tf.get_variable('we1_ffo', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_ffo', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_followed_followed = {
#            'encoder_h1': tf.get_variable('we1_ffed', shape=[n_input, n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h1': tf.get_variable('wd1_ffed', shape=[n_hidden_1, n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        weights_foursquare_concat = {
#            'encoder_h2': tf.get_variable('we1_fcc', shape=[n_hidden_1*7, n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_h2': tf.get_variable('wd1_fcc', shape=[n_hidden_2, n_hidden_1*7],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#
#        biases_foursquare_follow = {
#            'encoder_b1': tf.get_variable('be1_ff', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_ff', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_co_loc = {
#            'encoder_b1': tf.get_variable('be1_fcl', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_fcl', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_co_link = {
#            'encoder_b1': tf.get_variable('be1_fclk', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_fclk', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_follow_follow = {
#            'encoder_b1': tf.get_variable('be1_fff', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_fff', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_follow_in = {
#            'encoder_b1': tf.get_variable('be1_ffi', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_ffi', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_follow_out = {
#            'encoder_b1': tf.get_variable('be1_ffo', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_ffo', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_followed_followed = {
#            'encoder_b1': tf.get_variable('be1_ffed', shape=[n_hidden_1],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b1': tf.get_variable('bd1_ffed', shape=[n_input],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }
#        biases_foursquare_concat = {
#            'encoder_b2': tf.get_variable('be1_fcc', shape=[n_hidden_2],
#                initializer = tf.contrib.layers.xavier_initializer()),
#            'decoder_b2': tf.get_variable('bd1_fcc', shape=[n_hidden_1*7],
#                initializer = tf.contrib.layers.xavier_initializer())
#        }

        weights_twitter_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_co_loc = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_co_link = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_follow_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_follow_in = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_follow_out = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_followed_followed = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_twitter_concat = {
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1*7, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1*7])),
        }
        biases_twitter_follow = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_co_loc = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_co_link = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_follow_follow = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_follow_in = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_follow_out = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_followed_followed = {
            'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.zeros([n_input])),
        }
        biases_twitter_concat = {
            'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.zeros([n_hidden_1*7])),
        }


        weights_foursquare_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_co_loc = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_co_link = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_follow_follow = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_follow_in = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_follow_out = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_followed_followed = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        weights_foursquare_concat = {
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1*7, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1*7])),
        }
        biases_foursquare_follow = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_co_loc = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_co_link = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_follow_follow = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_follow_in = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_follow_out = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_followed_followed = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }
        biases_foursquare_concat = {
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1*7])),
        }

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_follow = tf.sigmoid(tf.add(tf.matmul(X2_follow,
                weights_foursquare_follow['encoder_h1']), biases_foursquare_follow['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_co_loc = tf.sigmoid(tf.add(tf.matmul(X2_co_loc,
                weights_foursquare_co_loc['encoder_h1']), biases_foursquare_co_loc['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_co_link = tf.sigmoid(tf.add(tf.matmul(X2_co_link,
                weights_foursquare_co_link['encoder_h1']), biases_foursquare_co_link['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_follow_follow = tf.sigmoid(tf.add(tf.matmul(X2_follow_follow,
                weights_foursquare_follow_follow['encoder_h1']), biases_foursquare_follow_follow['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_follow_in = tf.sigmoid(tf.add(tf.matmul(X2_follow_in,
                weights_foursquare_follow_in['encoder_h1']), biases_foursquare_follow_in['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_follow_out = tf.sigmoid(tf.add(tf.matmul(X2_follow_out,
                weights_foursquare_follow_out['encoder_h1']), biases_foursquare_follow_out['encoder_b1']))

        # Encoder Hidden foursquare_layer with relu activation #1
        foursquare_layer_1_followed_followed = tf.sigmoid(tf.add(tf.matmul(X2_followed_followed,
                weights_foursquare_followed_followed['encoder_h1']), biases_foursquare_followed_followed['encoder_b1']))

        # concat foursquare_layer 1
        foursquare_layer_1_concat = tf.concat([foursquare_layer_1_follow, foursquare_layer_1_co_loc, foursquare_layer_1_co_link, foursquare_layer_1_follow_follow, foursquare_layer_1_follow_in, foursquare_layer_1_follow_out, foursquare_layer_1_followed_followed], 1)
        # concat encoder hidden foursquare_layer with relu activation # 2
        foursquare_layer_2_encoder = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_1_concat,
                weights_foursquare_concat['encoder_h2']), biases_foursquare_concat['encoder_b2']))
        # concat encoder hidden foursquare_layer with relu activation # 2
        foursquare_layer_2_decoder = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_encoder,
                weights_foursquare_concat['decoder_h2']), biases_foursquare_concat['decoder_b2']))
        # split foursquare_layer 2 encoder result
        foursquare_layer_2_decoder_follow = foursquare_layer_2_decoder[:,0:n_hidden_1]
        foursquare_layer_2_decoder_co_loc = foursquare_layer_2_decoder[:,n_hidden_1:n_hidden_1*2]
        foursquare_layer_2_decoder_co_link = foursquare_layer_2_decoder[:,n_hidden_1*2:n_hidden_1*3]
        foursquare_layer_2_decoder_follow_follow = foursquare_layer_2_decoder[:,n_hidden_1*3:n_hidden_1*4]
        foursquare_layer_2_decoder_follow_in = foursquare_layer_2_decoder[:,n_hidden_1*4:n_hidden_1*5]
        foursquare_layer_2_decoder_follow_out = foursquare_layer_2_decoder[:,n_hidden_1*5:n_hidden_1*6]
        foursquare_layer_2_decoder_followed_followed = foursquare_layer_2_decoder[:,n_hidden_1*6:n_hidden_1*7]

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_follow = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_follow,
                weights_foursquare_follow['decoder_h1']), biases_foursquare_follow['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_co_loc = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_co_loc,
                weights_foursquare_co_loc['decoder_h1']), biases_foursquare_co_loc['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_co_link = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_co_link,
                weights_foursquare_co_link['decoder_h1']), biases_foursquare_co_link['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_follow_follow = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_follow_follow,
                weights_foursquare_follow_follow['decoder_h1']), biases_foursquare_follow_follow['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_follow_in = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_follow_in,
                weights_foursquare_follow_in['decoder_h1']), biases_foursquare_follow_in['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_follow_out = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_follow_out,
                weights_foursquare_follow_out['decoder_h1']), biases_foursquare_follow_out['decoder_b1']))

        # Decoder Hidden foursquare_layer with relu activation #3
        foursquare_layer_3_followed_followed = tf.sigmoid(tf.add(tf.matmul(foursquare_layer_2_decoder_followed_followed,
                weights_foursquare_followed_followed['decoder_h1']), biases_foursquare_followed_followed['decoder_b1']))


        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_follow = tf.sigmoid(tf.add(tf.matmul(X1_follow,
                weights_twitter_follow['encoder_h1']), biases_twitter_follow['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_co_loc = tf.sigmoid(tf.add(tf.matmul(X1_co_loc,
                weights_twitter_co_loc['encoder_h1']), biases_twitter_co_loc['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_co_link = tf.sigmoid(tf.add(tf.matmul(X1_co_link,
                weights_twitter_co_link['encoder_h1']), biases_twitter_co_link['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_follow_follow = tf.sigmoid(tf.add(tf.matmul(X1_follow_follow,
                weights_twitter_follow_follow['encoder_h1']), biases_twitter_follow_follow['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_follow_in = tf.sigmoid(tf.add(tf.matmul(X1_follow_in,
                weights_twitter_follow_in['encoder_h1']), biases_twitter_follow_in['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_follow_out = tf.sigmoid(tf.add(tf.matmul(X1_follow_out,
                weights_twitter_follow_out['encoder_h1']), biases_twitter_follow_out['encoder_b1']))

        # Encoder Hidden twitter_layer with relu activation #1
        twitter_layer_1_followed_followed = tf.sigmoid(tf.add(tf.matmul(X1_followed_followed,
                weights_twitter_followed_followed['encoder_h1']), biases_twitter_followed_followed['encoder_b1']))

        # concat twitter_layer 1
        twitter_layer_1_concat = tf.concat([twitter_layer_1_follow, twitter_layer_1_co_loc, twitter_layer_1_co_link, twitter_layer_1_follow_follow, twitter_layer_1_follow_in, twitter_layer_1_follow_out, twitter_layer_1_followed_followed], 1)
        # concat encoder hidden twitter_layer with relu activation # 2
        twitter_layer_2_encoder = tf.sigmoid(tf.add(tf.matmul(twitter_layer_1_concat,
                weights_twitter_concat['encoder_h2']), biases_twitter_concat['encoder_b2']))
        # concat encoder hidden twitter_layer with relu activation # 2
        twitter_layer_2_decoder = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_encoder,
                weights_twitter_concat['decoder_h2']), biases_twitter_concat['decoder_b2']))

        # split twitter_layer 2 encoder result
        twitter_layer_2_decoder_follow = twitter_layer_2_decoder[:,0:n_hidden_1]
        twitter_layer_2_decoder_co_loc = twitter_layer_2_decoder[:,n_hidden_1:n_hidden_1*2]
        twitter_layer_2_decoder_co_link = twitter_layer_2_decoder[:,n_hidden_1*2:n_hidden_1*3]
        twitter_layer_2_decoder_follow_follow = twitter_layer_2_decoder[:,n_hidden_1*3:n_hidden_1*4]
        twitter_layer_2_decoder_follow_in = twitter_layer_2_decoder[:,n_hidden_1*4:n_hidden_1*5]
        twitter_layer_2_decoder_follow_out = twitter_layer_2_decoder[:,n_hidden_1*5:n_hidden_1*6]
        twitter_layer_2_decoder_followed_followed = twitter_layer_2_decoder[:,n_hidden_1*6:n_hidden_1*7]

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_follow = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_follow,
                weights_twitter_follow['decoder_h1']), biases_twitter_follow['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_co_loc = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_co_loc,
                weights_twitter_co_loc['decoder_h1']), biases_twitter_co_loc['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_co_link = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_co_link,
                weights_twitter_co_link['decoder_h1']), biases_twitter_co_link['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_follow_follow = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_follow_follow,
                weights_twitter_follow_follow['decoder_h1']), biases_twitter_follow_follow['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_follow_in = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_follow_in,
                weights_twitter_follow_in['decoder_h1']), biases_twitter_follow_in['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_follow_out = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_follow_out,
                weights_twitter_follow_out['decoder_h1']), biases_twitter_follow_out['decoder_b1']))

        # Decoder Hidden twitter_layer with relu activation #3
        twitter_layer_3_followed_followed = tf.sigmoid(tf.add(tf.matmul(twitter_layer_2_decoder_followed_followed,
                weights_twitter_followed_followed['decoder_h1']), biases_twitter_followed_followed['decoder_b1']))

	x1_pred_follow = twitter_layer_3_follow
	x1_true_follow = X1_follow

	x1_pred_co_loc = twitter_layer_3_co_loc
	x1_true_co_loc = X1_co_loc

	x1_pred_co_link = twitter_layer_3_co_link
	x1_true_co_link = X1_co_link

	x1_pred_follow_follow = twitter_layer_3_follow_follow
	x1_true_follow_follow = X1_follow_follow

	x1_pred_follow_in = twitter_layer_3_follow_in
	x1_true_follow_in = X1_follow_in

	x1_pred_follow_out = twitter_layer_3_follow_out
	x1_true_follow_out = X1_follow_out

	x1_pred_followed_followed = twitter_layer_3_followed_followed
	x1_true_followed_followed = X1_followed_followed

        losses_1 = [tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_follow , x1_pred_follow) ,B1_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_co_loc , x1_pred_co_loc) ,B1_co_loc)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_co_link , x1_pred_co_link) ,B1_co_link)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_follow_follow , x1_pred_follow_follow) ,B1_follow_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_follow_in , x1_pred_follow_in) ,B1_follow_in)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_follow_out , x1_pred_follow_out) ,B1_follow_out)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x1_true_followed_followed , x1_pred_followed_followed) ,B1_followed_followed))
                 ]

	x2_pred_follow = foursquare_layer_3_follow
	x2_true_follow = X2_follow

	x2_pred_co_loc = foursquare_layer_3_co_loc
	x2_true_co_loc = X2_co_loc

	x2_pred_co_link = foursquare_layer_3_co_link
	x2_true_co_link = X2_co_link

	x2_pred_follow_follow = foursquare_layer_3_follow_follow
	x2_true_follow_follow = X2_follow_follow

	x2_pred_follow_in = foursquare_layer_3_follow_in
	x2_true_follow_in = X2_follow_in

	x2_pred_follow_out = foursquare_layer_3_follow_out
	x2_true_follow_out = X2_follow_out

	x2_pred_followed_followed = foursquare_layer_3_followed_followed
	x2_true_followed_followed = X2_followed_followed

        losses_2 = [tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_follow , x2_pred_follow) ,B2_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_co_loc , x2_pred_co_loc) ,B2_co_loc)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_co_link , x2_pred_co_link) ,B2_co_link)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_follow_follow , x2_pred_follow_follow) ,B2_follow_follow)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_follow_in , x2_pred_follow_in) ,B2_follow_in)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_follow_out , x2_pred_follow_out) ,B2_follow_out)),
                  tf.nn.l2_loss(tf.multiply(tf.subtract(x2_true_followed_followed , x2_pred_followed_followed) ,B2_followed_followed))
                 ]

        losses_3 = tf.nn.l2_loss(twitter_layer_2_encoder - foursquare_layer_2_encoder)

        losses_4 = [
                    0.5 * tf.nn.l2_loss(weights_twitter_follow['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_co_loc['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_co_loc['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_co_link['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_co_link['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_follow['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_follow['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_in['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_in['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_out['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_follow_out['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_followed_followed['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_followed_followed['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_twitter_concat['encoder_h2']),
                    0.5 * tf.nn.l2_loss(weights_twitter_concat['encoder_h2']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_co_loc['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_co_loc['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_co_link['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_co_link['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_follow['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_follow['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_in['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_in['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_out['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_follow_out['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_followed_followed['encoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_followed_followed['decoder_h1']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_concat['encoder_h2']),
                    0.5 * tf.nn.l2_loss(weights_foursquare_concat['encoder_h2'])
                ]

        cost_1 = tf.add_n(losses_1)
        cost_2 = tf.add_n(losses_2)
        cost_3 = losses_3
        cost_4 = tf.add_n(losses_4)
        alpha = 1
        beta = 1
        theata = 0.02
        cost = cost_1 + cost_2 * alpha + cost_3 * beta + cost_4 * theata
	#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

    	sess.run(init)
        pre_echos_completed = 0
        while(self._epochs_completed < training_epochs):
            batch_x1_follow, batch_x1_co_loc, batch_x1_co_link, batch_x1_follow_follow, batch_x1_follow_in, batch_x1_follow_out, batch_x1_followed_followed, batch_x2_follow, batch_x2_co_loc, batch_x2_co_link, batch_x2_follow_follow, batch_x2_follow_in, batch_x2_follow_out, batch_x2_followed_followed, batch_penalty1_follow, batch_penalty1_co_loc, batch_penalty1_co_link, batch_penalty1_follow_follow, batch_penalty1_follow_in, batch_penalty1_follow_out, batch_penalty1_followed_followed, batch_penalty2_follow, batch_penalty2_co_loc, batch_penalty2_co_link, batch_penalty2_follow_follow, batch_penalty2_follow_in, batch_penalty2_follow_out, batch_penalty2_followed_followed = self.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X1_follow: batch_x1_follow, X1_co_loc: batch_x1_co_loc, X1_co_link: batch_x1_co_link, X1_follow_follow: batch_x1_follow_follow, X1_follow_in: batch_x1_follow_in, X1_follow_out: batch_x1_follow_out, X1_followed_followed: batch_x1_followed_followed, X2_follow: batch_x2_follow, X2_co_loc: batch_x2_co_loc, X2_co_link: batch_x2_co_link, X2_follow_follow: batch_x2_follow_follow, X2_follow_in: batch_x2_follow_in, X2_follow_out: batch_x2_follow_out, X2_followed_followed: batch_x2_followed_followed, B1_follow: batch_penalty1_follow, B1_co_loc: batch_penalty1_co_loc, B1_co_link: batch_penalty1_co_link, B1_follow_follow: batch_penalty1_follow_follow, B1_follow_in: batch_penalty1_follow_in, B1_follow_out: batch_penalty1_follow_out, B1_followed_followed: batch_penalty1_followed_followed, B2_follow: batch_penalty2_follow, B2_co_loc: batch_penalty2_co_loc, B2_co_link: batch_penalty2_co_link, B2_follow_follow: batch_penalty2_follow_follow, B2_follow_in: batch_penalty2_follow_in, B2_follow_out: batch_penalty2_follow_out, B2_followed_followed: batch_penalty2_followed_followed})
#            x1 = sess.run(X1_follow, feed_dict={X1_follow: batch_x1_follow})
#            p1 = sess.run(B1_follow, feed_dict={B1_follow: batch_penalty1_follow})
#            x2 = sess.run(X2_follow, feed_dict={X2_follow: batch_x2_follow})
#            p2 = sess.run(B2_follow, feed_dict={B2_follow: batch_penalty2_follow})
#            print ' '.join(str(t) for t in x1[1])
#            print ' '.join(str(t) for t in p1[1])
#            print ' '.join(str(t) for t in x2[1])
#            print ' '.join(str(t) for t in p2[1])
#            for i in range(10):
#                for j in range(len(x1[0])):
#                    if x1[i][j] != 0:
#                        print str(i) + '\t' + str(j) + '\t' + str(x1[i][j]) + '\t' + str(p1[i][j])
#                    if x2[i][j] != 0:
#                        print str(i) + '\t' + str(j) + '\t' + str(x2[i][j]) + '\t' + str(p2[i][j])
#            break
            ''' Display logs per epoch step'''
            if self._epochs_completed > pre_echos_completed:
                pre_echos_completed += 1
                print("Epoch:", '%04d' % (self._epochs_completed), "cost=", "{:.9f}".format(c))
                if pre_echos_completed > 1000:
                    feature = sess.run(foursquare_layer_2_encoder, feed_dict={X1_follow: twitter_follow, X1_co_loc: twitter_co_loc, X1_co_link: twitter_co_link, X1_follow_follow: twitter_follow_follow, X1_follow_in: twitter_follow_in, X1_follow_out: twitter_follow_out, X1_followed_followed: twitter_followed_followed, X2_follow: foursquare_follow, X2_co_loc: foursquare_co_loc, X2_co_link: foursquare_co_link, X2_follow_follow: foursquare_follow_follow, X2_follow_in: foursquare_follow_in, X2_follow_out: foursquare_follow_out, X2_followed_followed: foursquare_followed_followed})
                    result = expResult('a','b')
                    result.run(feature)


#   	total_batch = int(n_input/batch_size)
#    	# Training cycle
#    	for epoch in range(training_epochs):
#            # Loop over all batches
#            for i in range(total_batch):
#
#                # generate batch
#                batch_index = self.generate_batch(instance_num, batch_size)
#
#                batch_x1_follow = twitter_follow[batch_index]
#                batch_x1_co_loc = twitter_co_loc[batch_index]
#                batch_x1_co_link = twitter_co_link[batch_index]
#                batch_x1_follow_follow = twitter_follow_follow[batch_index]
#                batch_x1_follow_in = twitter_follow_in[batch_index]
#                batch_x1_follow_out = twitter_follow_out[batch_index]
#                batch_x1_followed_followed = twitter_followed_followed[batch_index]
#
#                batch_x2_follow = foursquare_follow[batch_index]
#                batch_x2_co_loc = foursquare_co_loc[batch_index]
#                batch_x2_co_link = foursquare_co_link[batch_index]
#                batch_x2_follow_follow = foursquare_follow_follow[batch_index]
#                batch_x2_follow_in = foursquare_follow_in[batch_index]
#                batch_x2_follow_out = foursquare_follow_out[batch_index]
#                batch_x2_followed_followed = foursquare_followed_followed[batch_index]
#
#                # Run optimization op (backprop) and cost op (to get loss value)
#                _, c = sess.run([optimizer, cost], feed_dict={X1_follow: batch_x1_follow, X1_co_loc: batch_x1_co_loc, X1_co_link: batch_x1_co_link, X1_follow_follow: batch_x1_follow_follow, X1_follow_in: batch_x1_follow_in, X1_follow_out: batch_x1_follow_out, X1_followed_followed: batch_x1_followed_followed, X2_follow: batch_x2_follow, X2_co_loc: batch_x2_co_loc, X2_co_link: batch_x2_co_link, X2_follow_follow: batch_x2_follow_follow, X2_follow_in: batch_x2_follow_in, X2_follow_out: batch_x2_follow_out, X2_followed_followed: batch_x2_followed_followed})
#            # Display logs per epoch step
#            if epoch % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
#                feature = sess.run(foursquare_layer_2_encoder, feed_dict={X1_follow: twitter_follow, X1_co_loc: twitter_co_loc, X1_co_link: twitter_co_link, X1_follow_follow: twitter_follow_follow, X1_follow_in: twitter_follow_in, X1_follow_out: twitter_follow_out, X1_followed_followed: twitter_followed_followed, X2_follow: foursquare_follow, X2_co_loc: foursquare_co_loc, X2_co_link: foursquare_co_link, X2_follow_follow: foursquare_follow_follow, X2_follow_in: foursquare_follow_in, X2_follow_out: foursquare_follow_out, X2_followed_followed: foursquare_followed_followed})
#                if epoch > 400:
#                    result = expResult('a','b')
#                    result.run(feature)
        print("Optimization Finished!")

        # output feature vector
        feature = sess.run(foursquare_layer_2_encoder, feed_dict={X1_follow: twitter_follow, X1_co_loc: twitter_co_loc, X1_co_link: twitter_co_link, X1_follow_follow: twitter_follow_follow, X1_follow_in: twitter_follow_in, X1_follow_out: twitter_follow_out, X1_followed_followed: twitter_followed_followed, X2_follow: foursquare_follow, X2_co_loc: foursquare_co_loc, X2_co_link: foursquare_co_link, X2_follow_follow: foursquare_follow_follow, X2_follow_in: foursquare_follow_in, X2_follow_out: foursquare_follow_out, X2_followed_followed: foursquare_followed_followed})
        # dump embedding feature into file
        fp = open('embedding_feature','w')
        for i in range(len(feature)):
            fp.write(' '.join(str(x) for x in feature[i])+'\n')
        fp.close()

        sess.close()

        return feature
