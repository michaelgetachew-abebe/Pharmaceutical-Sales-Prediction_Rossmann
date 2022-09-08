import numpy as np

def mae(y, yhat):
    '''
    mae: calculates the mean absoulute error of the model given
        y: the true value
        yhat: the predicted value
    '''
    mae = np.abs(np.subtract(y,yhat)).mean()
    return mae
    
def mse(y, yhat):
    '''
    mse: calcualtes the mean squraed error of the model given
        y: the true value
        yhat: the predicted value
    '''
    mse = np.square(np.subtract(y,yhat)).mean()
    return mse

def rmse(y, yhat):
    '''
    rmse: calcualtes the root mean squared error of the model given 
        y: the true value
        yhat: the predicted value
    '''
    rmspe = np.sqrt(np.mean((y - yhat)**2))
    return rmspe