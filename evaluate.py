import abc

#this is the class evapuate, it is used to evalute the performance of the classifiers

class evaluate:
    """ evaluate class """
    #this is the attributes of the class evaluate
    name = ''
    evaluateDescription = ''
    
    evaluateValue = 0
    evaluateStatics = 0
    debug = False
    
    #this is the method of class evaluate
    def __init__(self, name):
    	self.name = name

    @abc.abstractmethod
    def evaluate(self, predLabel, testLabel):
    	return

