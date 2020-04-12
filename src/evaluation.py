import numpy as np
import pandas as pd

def map5eval(preds, dtrain, k=5):
    actual = dtrain.get_label()
    predicted = (-preds).argsort(axis=1)[:,:k]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', metric

