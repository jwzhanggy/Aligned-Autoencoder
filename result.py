import abc

class result:
    """ this is the result class """
    #attributes of class result
    @abc.abstractmethod
    def save(self):
    	return

    @abc.abstractmethod
    def load(self):
        return

