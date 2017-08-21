from profile import *
from expDataset import expDataset
from expSetting import expSetting
from expMethod import expMethod
from expResult import expResult
import numpy as np
from scipy import interp
import decimal
import pylab as pl
from pylab import *
import random
from sklearn.metrics import roc_curve, auc
import sys
import pickle
import os,sys
import time

dataset = expDataset("", "")
#dataset.prefix_path = '../data/link_data'
#dataset.dataset_list = ['foursquare', 'twitter']

setting = expSetting()
method = expMethod()
result = expResult()

pro = profile(dataset, setting, method, result)
print "All Start"
start = time.clock()
pro.run_baseline()
elapsed = (time.clock() - start)
print("Time used:", elapsed)
print "All Finish"

