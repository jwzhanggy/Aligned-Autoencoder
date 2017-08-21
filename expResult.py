from result import result
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# this is my exp result class, it inherits from class result
class expResult(result):
    path_name_prefix = '../data/link_data'
    user_dict_path = path_name_prefix + '/foursquare_user_index'
    result_file = 'results/sample_result_0.1'
    file_train_ori = ''
    file_test_ori = ''

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    user_dict = {}
    embedding_dict = {}

    def open_user_dict(self):
        for line in open(self.user_dict_path):
            arr = line.strip().split('\t')
            self.user_dict[arr[0]] = arr[1]

    def open_embedding_feature(self, embedding_feature):
        index = 0
        for line in embedding_feature:
            user = self.user_dict[str(index)]
            self.embedding_dict[user] = line.tolist()
            index += 1

    def prepare_data(self, embedding_feature):
        self.open_user_dict()
        self.open_embedding_feature(embedding_feature)

        for line in open(self.file_train_ori):
            arr = line.strip().split('\t')
            u1 = arr[0]
            u2 = arr[1]
            self.y_train.append(int(arr[2]))

            x_feature = []
            if self.embedding_dict.has_key(u1):
                x_feature = self.embedding_dict[u1]
            else:
                x_feature = np.zeros(50).tolist()
            if self.embedding_dict.has_key(u2):
                x_feature = x_feature + self.embedding_dict[u2]
            else:
                x_feature = x_feature + np.zeros(50).tolist()
            self.x_train.append(x_feature)

        for line in open(self.file_test_ori):
            arr = line.strip().split('\t')
            u1 = arr[0]
            u2 = arr[1]
            self.y_test.append(int(arr[2]))

            x_feature = []
            if self.embedding_dict.has_key(u1):
                x_feature = self.embedding_dict[u1]
            else:
                x_feature = np.zeros(50).tolist()
            if self.embedding_dict.has_key(u2):
                x_feature = x_feature + self.embedding_dict[u2]
            else:
                x_feature = x_feature + np.zeros(50).tolist()
            self.x_test.append(x_feature)

    def run(self, embedding_feature, fold, sample_rate):
        self.result_file = 'results/sample_result_'+str(sample_rate)
        self.file_train_ori = self.path_name_prefix + '/kfolds/sample_'+str(fold)+'_'+str(sample_rate)
        self.file_test_ori = self.path_name_prefix + '/kfolds/test_'+str(fold)
        self.prepare_data(embedding_feature)

#        # dump train_file
#        ftr = open('./train_file', 'w')
#        for i in range(len(self.y_train)):
#            ftr.write(str(self.y_train[i])+'\t'+' '.join(str(x) for x in self.x_train[i]) + '\n')
#
#        # dump test_file
#        ftr = open('./test_file', 'w')
#        for i in range(len(self.y_test)):
#            ftr.write(str(self.y_test[i])+'\t'+' '.join(str(x) for x in self.x_test[i]) + '\n')

        #clf = SVC(kernel='linear', probability=True, cache_size=4000)
        clf = LinearSVC()

        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        #test_prob = clf.predict_proba(self.x_test)
        #pos_prob = test_prob[:,1]
        y_score = clf.decision_function(self.x_test)

        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        #auc = roc_auc_score(self.y_test, pos_prob)
        auc = roc_auc_score(self.y_test, y_score)

        fp = open(self.result_file, 'a')
        fp.write(str(precision)+'\t'+str(recall)+'\t'+str(f1)+'\t'+str(accuracy)+'\t'+str(auc)+'\n')

#        # dump predict result
#        y = clf.predict(self.x_test)
#        fp = open('test_predict','w')
#        for i in range(len(y)):
#           fp.write(str(y[i]) + '\t' + str(self.y_test[i]) + '\n')
