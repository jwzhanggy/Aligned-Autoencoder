import abc

#-----------------------------------------------------
class setting:
    '''this is the class setting, if will incorporate the classifler,
    evaluation emthod, result destination together and do some environemnt setting work'''

    @abc.abstractmethod
    def load_classify_evaluate_save(self):
        return

