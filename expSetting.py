from setting import *
import math
from sklearn import cross_validation
import copy
import random
import pickle

# this is the exp setting class, if will do something real!
class expSetting(setting):

    fold = 1
    sample_rate = 0.1
    setting_file = 'setting_file'

    def load(self):
        for line in open(self.setting_file):
            arr = line.strip().split(' ')
            if arr[0] == 'fold':
                self.fold = arr[1]
            if arr[0] == 'sample_rate':
                self.sample_rate = arr[1]
