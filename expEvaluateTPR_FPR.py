from evaluate import *
import numpy
from sklearn.metrics import roc_curve, auc
import pylab as pl
# this is my exp evaluate class, it inherits from class dataset
class expEvaluateTPR_FPR(evaluate):
    
    def evaluate(self, real_label, confidence):
        fpr, tpr, thresholds = roc_curve(real_label, confidence)
#        
#        conf_label_dict = {}
#        for label, conf in zip(real_label, confidence):
#            if conf not in conf_label_dict:
#                conf_label_dict[conf] = [label]
#            else:
#                conf_label_dict[conf].append(label)
#        
#        sorted_conf_list = sorted(confidence, cmp = self.reverse_numeric)
#        count = 0
#        true_label_count = 0
#        index = 0
#        
#        while count < 30:
#            label_list = conf_label_dict[sorted_conf_list[index]]
#            index = index + 1
#            for label in label_list:
#                if label == 1:
#                    true_label_count = true_label_count + 1
#                count = count + 1
#                if count == 30:
#                    break
#        if true_label_count > 30:
#            print true_label_count
#            print count
#        precision_at_30 = float(true_label_count) / 30
        #return {'tpr': tpr, 'fpr': fpr, 'precision@30': precision_at_30}
        return {'tpr': tpr, 'fpr': fpr}
    
    def reverse_numeric(self, x, y):
        if x > y:
            return -1
        elif y > x:
            return 1
        else:
            return 0