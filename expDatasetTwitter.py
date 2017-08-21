import os.path
from os import listdir
import numpy as np
from dataset import dataset
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from random import sample


'''
Class of expDatatwitter
used to load data from hard disk
'''
class expDatasetTwitter(dataset):
    #=========== file path =============
    #dataset_prefix = '../data/test_data'
    dataset_prefix = '../data/link_data'

    # inputs
    file_twitter_user = dataset_prefix+'/twitter_user'
    file_twitter_co_loc = dataset_prefix+'/twitter_co_locations'
    file_twitter_contact_link = dataset_prefix+'/twitter_contact_links'
    file_twitter_follow_link = dataset_prefix+'/twitter_follow_links'

    # outputs
    file_twitter_user_index = dataset_prefix+'/twitter_user_index'
    file_twitter_co_loc_matrix = dataset_prefix+'/matrix/twitter_co_loc_matrix'
    file_twitter_co_link_matrix = dataset_prefix+'/matrix/twitter_co_link_matrix'
    file_twitter_follow_matrix = dataset_prefix+'/matrix/twitter_follow_matrix'
    file_twitter_follow_follow_matrix = dataset_prefix+'/matrix/twitter_follow_follow_matrix'
    file_twitter_follow_in_matrix = dataset_prefix+'/matrix/twitter_follow_in_matrix'
    file_twitter_follow_out_matrix = dataset_prefix+'/matrix/twitter_follow_out_matrix'
    file_twitter_followed_followed_matrix = dataset_prefix+'/matrix/twitter_followed_followed_matrix'

    #=========== variables =============
    twitter_dict = {}
    twitter_user_num = 0

    #=========== load users =============
    def load_dict(self):
        if os.path.isfile(self.file_twitter_user_index) and os.stat(self.file_twitter_user_index).st_size != 0:
            print "exists user_index file"
            for line in open(self.file_twitter_user_index):
                arr = line.strip().split('\t')
                if len(arr) != 2:
                    continue
                self.twitter_dict[arr[1]] = int(arr[0])
                self.twitter_user_num += 1

        else:
            twitter_index = open(self.file_twitter_user_index, 'w')
            for line in open(self.file_twitter_user):
                user_id = str(line.strip())
                twitter_index.write(str(self.twitter_user_num) + '\t' + user_id + '\n')
	        self.twitter_dict[user_id] = self.twitter_user_num
                self.twitter_user_num += 1
            twitter_index.close()

    def normalize(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X)
        return X_train_minmax

    #=========== trans_matrix =============
    def trans_matrix(self, read_file_name, dump_file_name):
        matrix = dok_matrix((self.twitter_user_num, self.twitter_user_num))
        for line in open(read_file_name):
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue
            user_1_index = 0
            user_2_index = 0
            if self.twitter_dict.has_key(arr[0]):
                user_1_index = self.twitter_dict[arr[0]]
            if self.twitter_dict.has_key(arr[1]):
                user_2_index = self.twitter_dict[arr[1]]
            weight = arr[2]
            matrix[user_1_index, user_2_index] = weight

        matrix = self.normalize(matrix.toarray())

        # dump matrix
        if not os.path.isfile(dump_file_name):
            np.savetxt(dump_file_name, matrix, fmt='%.9f')
        return matrix

    #======================== load function ========================
    #used to call each specific functions to load data
    def load(self):
        print "loading twitter dataset"

        # load dict
        self.load_dict()
        print "twitter_user_number : " + str(self.twitter_user_num)

        #trans matrix
        print "begin trans matrxi"
        twitter_co_loc = self.trans_matrix(self.file_twitter_co_loc, self.file_twitter_co_loc_matrix)
        print "trans twitter_co_loc"
        twitter_co_link = self.trans_matrix(self.file_twitter_contact_link, self.file_twitter_co_link_matrix)
        print "trans twitter_co_link"
        twitter_follow = self.trans_matrix(self.file_twitter_follow_link, self.file_twitter_follow_matrix)
        print "basic matrix"

        # twitter follow
        twitter_follow_follow = self.normalize(twitter_follow * twitter_follow)
        twitter_follow_in = self.normalize(twitter_follow * np.transpose(twitter_follow))
        twitter_follow_out = self.normalize(np.transpose(twitter_follow) * twitter_follow)
        twitter_followed_followed = self.normalize(np.transpose(twitter_follow) * np.transpose(twitter_follow))
        print "four follow matrix"

        # dump follow matrix
        if not os.path.isfile(self.file_twitter_follow_follow_matrix):
            np.savetxt(self.file_twitter_follow_follow_matrix, twitter_follow_follow, fmt='%f')
            np.savetxt(self.file_twitter_follow_in_matrix, twitter_follow_in, fmt='%f')
            np.savetxt(self.file_twitter_follow_out_matrix, twitter_follow_out, fmt='%f')
            np.savetxt(self.file_twitter_followed_followed_matrix, twitter_followed_followed, fmt='%f')
        print "dump matrix done"

        twitter_network_data = {'follow_link': twitter_follow, 'co_loc': twitter_co_loc, 'contact_link': twitter_co_link, 'follow_follow': twitter_follow_follow, 'follow_in': twitter_follow_in, 'follow_out': twitter_follow_out, 'followed_followed': twitter_followed_followed}

        print "twitter dataset has been loaded"
        return twitter_network_data
