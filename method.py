import abc


class method:
    '''this is the classifier method'''
    methodName = ''
    methodType = ''
    methodDescription = ''
    
    methodTime = 0
    methodTrainTime = 0
    methodTestTime = 0
    debug = False

    #function of class method
    def __init__(self, name, mType):
    	self.methodName = name
    	self.methodType = mType

    @abc.abstractmethod
    def classify(self, trainData, trainLabel, testData):
    	return

