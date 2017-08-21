import abc




#-------------------------------------------------------------------
class dataset:
    """ Dataset: Abstract Class """
    name = ''
    dataDescrition = ''
    debug = False
    
    # constructive function
    def __init__(self, dName, dDescription):
    	self.dataName = dName
    	self.dataDescription = dDescription

    @abc.abstractmethod
    def load(self):
    	return
 
 
