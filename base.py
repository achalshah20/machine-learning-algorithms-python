class LinearModel(object):

    def train(self,X,y):
        raise NotImplementedError('subclasses must override train()!')

    def predict(self,X):
        raise NotImplementedError('subclasses must override predict()!')
