from expDatasetTwitter import expDatasetTwitter
from expDatasetFoursquare import expDatasetFoursquare
from dataset import dataset
from os import listdir
from random import sample
from random import shuffle
import numpy as np
from sklearn.model_selection import KFold

# this is my exp data set, it inherits from class dataset

class expDataset(dataset):
    dataset_list = ['foursquare', 'twitter']
    prefix_path = '../data/link_data'
    #prefix_path = '../data/test_data'
    kfolds_dir = prefix_path + '/kfolds'
    file_kfolds_train = kfolds_dir + '/train'
    file_kfolds_test = kfolds_dir + '/test'
    file_sample = kfolds_dir + '/sample'
    file_foursquare_user = prefix_path + '/foursquare_user_index'
    file_foursquare_follow_link = prefix_path + '/foursquare_follow_links'
    pos_neg_rate = 1.0

    def prepare(self):
        print "prepare cross validata and sample data"

        user_dict = {}
        inverse_user_dict = {}
        # load foursquare user
        for line in open(self.file_foursquare_user):
            arr = line.strip().split('\t')
            user_dict[arr[1]] = int(arr[0])
            inverse_user_dict[int(arr[0])] = arr[1]
        user_num = len(user_dict.keys())
        print user_num

        matrix = np.zeros((user_num, user_num), dtype=np.int32)
        # load twitter follow : positive data
        for line in open(self.file_foursquare_follow_link):
            line = line.strip()
            arr = line.split('\t')
            u1 = arr[0]
            u2 = arr[1]
            if user_dict.has_key(u1) and user_dict.has_key(u2):
                u1_idx = user_dict[u1]
                u2_idx = user_dict[u2]
                matrix[u1_idx][u2_idx] = 1

        # generate positive and negative data
        print "generate big matrix"
        pos_list = []
        neg_list = []
        for i in range(user_num):
            for j in range(user_num):
                if (i != j):
                    if matrix[i][j] == 1: # positive sample
                        pos_list.append((i,j,1))
                    elif matrix[i][j] == 0: # negative sample
                        neg_list.append((i,j,0))
                    else:
                        print "error"

        # sample negative data
        print "sample negative data"
        pos_num = len(pos_list)
        neg_num = int(pos_num * self.pos_neg_rate)
        print "pos_num " + str(pos_num)
        print "neg_num " + str(neg_num)

        neg_idx = [x for x in range(len(neg_list))]
        neg_sample_index = sample(neg_idx, neg_num)
        neg_data = np.array(neg_list)
        neg_sample_data = (neg_data[neg_sample_index]).tolist()
        data = pos_list
        data.extend(neg_sample_data)
        shuffle(data)
        print "total " + str(len(data))

        # cross validation
        kf = KFold(n_splits=10)

        index = 1
        for train, test in kf.split(data):
            file_train = open(self.file_kfolds_train + '_' + str(index), 'w')
            file_test = open(self.file_kfolds_test + '_' + str(index), 'w')

            file_train.write('\n'.join(str(inverse_user_dict[data[x][0]]+'\t'+inverse_user_dict[data[x][1]]+'\t'+str(data[x][2])) for x in train))
            file_test.write('\n'.join(str(inverse_user_dict[data[x][0]]+'\t'+inverse_user_dict[data[x][1]]+'\t'+str(data[x][2])) for x in test))

            # random sample follow links
            for sample_ratio in np.arange(0.1,1.0,0.1):
                print "fold " + str(index) + " sample_raito " + str(sample_ratio)
                file_train_sample = open(self.file_sample + '_' + str(index) + '_' + str(sample_ratio), 'w')
                sample_num = int(len(train) * sample_ratio)
                sample_index = sample(train, sample_num)
                file_train_sample.write('\n'.join(str(inverse_user_dict[data[x][0]]+'\t'+inverse_user_dict[data[x][1]]+'\t'+str(data[x][2])) for x in sample_index))
            # sample rate 1.0
            file_train_sample = open(self.file_sample + '_' + str(index) + '_' + str(1.0), 'w')
            file_train_sample.write('\n'.join(str(inverse_user_dict[data[x][0]]+'\t'+inverse_user_dict[data[x][1]]+'\t'+str(data[x][2])) for x in train))

            index += 1

    def load(self, fold, sample_rate):

        if listdir(self.kfolds_dir) == []:
            self.prepare()

        expDatasetTwitter_obj = expDatasetTwitter('Twitter', '')
        expDatasetFoursquare_obj = expDatasetFoursquare('Foursquare', '')

        twitter_network_data = expDatasetTwitter_obj.load()
        foursquare_network_data = expDatasetFoursquare_obj.load(fold, sample_rate)

        network = {'foursquare': foursquare_network_data, 'twitter': twitter_network_data}
        return network
