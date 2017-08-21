from evaluate import *
from sklearn.metrics import *
from copy import *

# this is my exp evaluate class, it inherits from class dataset
class expEvaluateF1_Precision_Recall_Accuracy(evaluate):

    def evaluate(self, y_pred, y_true):
        count = 0
        for pred, true in zip(y_pred, y_true):
            if pred == true:
                count += 1
        
        accuracy = float(count) / len(y_true)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
#        conf_copy = deepcopy(confidence)
#        conf_copy.sort(reverse = True)
#        top30 = conf_copy[:30]
#        count = 0
#        count_of_30 = 0
#        for y, conf in zip(y_true, confidence):
#            if conf in top30:
#                if count_of_30 == 30:
#                    break
#                count_of_30 = count_of_30 + 1
#                if y == 1:
#                    count = count + 1
#        print count_of_30
#        print count
#        precision_top30 = (float(count) / float(30))
        recall = recall_score(y_true, y_pred)
        return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}