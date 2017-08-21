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
Class of expDatafoursquare
used to load data from hard disk
'''
class expDatasetFoursquare(dataset):
    #=========== file path =============
    #dataset_prefix = '../data/test_data'
    dataset_prefix = '../data/link_data'

    # inputs
    file_foursquare_user = dataset_prefix+'/foursquare_user'
    file_foursquare_co_loc = dataset_prefix+'/foursquare_co_list_locations'
    file_foursquare_contact_link = dataset_prefix+'/foursquare_co_tip_locations'
    file_foursquare_follow_link = dataset_prefix+'/foursquare_follow_links'

    # outputs
    file_foursquare_user_index = dataset_prefix+'/foursquare_user_index'
    file_foursquare_co_loc_matrix = dataset_prefix+'/matrix/foursquare_co_list_loc_matrix'
    file_foursquare_co_link_matrix = dataset_prefix+'/matrix/foursquare_co_tip_loc_matrix'
    file_foursquare_follow_matrix = dataset_prefix+'/matrix/foursquare_follow_matrix'
    file_foursquare_follow_follow_matrix = dataset_prefix+'/matrix/foursquare_follow_follow_matrix'
    file_foursquare_follow_in_matrix = dataset_prefix+'/matrix/foursquare_follow_in_matrix'
    file_foursquare_follow_out_matrix = dataset_prefix+'/matrix/foursquare_follow_out_matrix'
    file_foursquare_followed_followed_matrix = dataset_prefix+'/matrix/foursquare_followed_followed_matrix'

    #=========== variables =============
    foursquare_dict = {}
    foursquare_user_num = 0

    #=========== load users =============
    def load_dict(self):
        if os.path.isfile(self.file_foursquare_user_index) and os.stat(self.file_foursquare_user_index).st_size != 0:
            print "exists user_index file"
            for line in open(self.file_foursquare_user_index):
                arr = line.strip().split('\t')
                if len(arr) != 2:
                    continue
                self.foursquare_dict[arr[1]] = int(arr[0])
                self.foursquare_user_num += 1

        else:
            foursquare_index = open(self.file_foursquare_user_index, 'w')
            for line in open(self.file_foursquare_user):
                user_id = str(line.strip())
                foursquare_index.write(str(self.foursquare_user_num) + '\t' + user_id + '\n')
	        self.foursquare_dict[user_id] = self.foursquare_user_num
                self.foursquare_user_num += 1
            foursquare_index.close()

    def normalize(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X)
        return X_train_minmax

    #=========== trans_matrix =============
    def trans_matrix(self, read_file_name, dump_file_name):
        matrix = dok_matrix((self.foursquare_user_num, self.foursquare_user_num))
        for line in open(read_file_name):
            arr = line.strip().split('\t')
            if len(arr) != 3:
                continue
            user_1_index = 0
            user_2_index = 0
            if self.foursquare_dict.has_key(arr[0]):
                user_1_index = self.foursquare_dict[arr[0]]
            if self.foursquare_dict.has_key(arr[1]):
                user_2_index = self.foursquare_dict[arr[1]]
            weight = arr[2]
            matrix[user_1_index, user_2_index] = weight

        matrix = self.normalize(matrix.toarray())

        # dump matrix
        if not os.path.isfile(dump_file_name):
            np.savetxt(dump_file_name, matrix, fmt='%.9f')
        return matrix

    #======================== load function ========================
    #used to call each specific functions to load data
    def load(self, fold, sample_rate):
        print "loading foursquare dataset"

        # sample data
        self.file_foursquare_follow_link =  self.dataset_prefix + '/kfolds/sample_' + str(fold) + '_' + str(sample_rate)
        self.file_foursquare_co_loc = self.dataset_prefix+'/sample/foursquare_co_list_locations'
        self.file_foursquare_contact_link = self.dataset_prefix+'/sample/foursquare_co_tip_locations'

        # load dict
        self.load_dict()
        print "foursquare_user_number : " + str(self.foursquare_user_num)

        #trans matrix
        foursquare_co_loc = self.trans_matrix(self.file_foursquare_co_loc, self.file_foursquare_co_loc_matrix)
        foursquare_co_link = self.trans_matrix(self.file_foursquare_contact_link, self.file_foursquare_co_link_matrix)
        foursquare_follow = self.trans_matrix(self.file_foursquare_follow_link, self.file_foursquare_follow_matrix)

        # foursquare follow
        foursquare_follow_follow = self.normalize(foursquare_follow * foursquare_follow)
        foursquare_follow_in = self.normalize(foursquare_follow * np.transpose(foursquare_follow))
        foursquare_follow_out = self.normalize(np.transpose(foursquare_follow) * foursquare_follow)
        foursquare_followed_followed = self.normalize(np.transpose(foursquare_follow) * np.transpose(foursquare_follow))

        # dump follow matrix
        if not os.path.isfile(self.file_foursquare_follow_follow_matrix):
            np.savetxt(self.file_foursquare_follow_follow_matrix, foursquare_follow_follow, fmt='%f')
            np.savetxt(self.file_foursquare_follow_in_matrix, foursquare_follow_in, fmt='%f')
            np.savetxt(self.file_foursquare_follow_out_matrix, foursquare_follow_out, fmt='%f')
            np.savetxt(self.file_foursquare_followed_followed_matrix, foursquare_followed_followed, fmt='%f')

        foursquare_network_data = {'follow_link': foursquare_follow, 'co_list_loc': foursquare_co_loc, 'co_tip_loc': foursquare_co_link, 'follow_follow': foursquare_follow_follow, 'follow_in': foursquare_follow_in, 'follow_out': foursquare_follow_out, 'followed_followed': foursquare_followed_followed}

        print "foursquare dataset has been loaded"
        return foursquare_network_data
