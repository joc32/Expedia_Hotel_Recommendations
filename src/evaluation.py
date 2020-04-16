import numpy as np
import pandas as pd

def map5eval(predicted, actual, k=5):
    #actual = dtrain.get_label()
    #predicted = (-preds).argsort(axis=1)[:,:k]
    y = np.array([np.array(xi) for xi in predicted])
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==y[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', metric

def random_recommendations(df):
    df['random'] = pd.Series(index=df.index, dtype=object)

    for i, row in df.iterrows():
        df.at[i, 'random'] = np.random.randint(0,101,5)

    map5_score = map5eval(df['random'].values, df['hotel_cluster'].values)
    print(map5_score)

def get_user_cluster():
    print('NOT IMPLEMENTED')
    exit()

def get_decision_tree():
    print('NOT IMPLEMENTED')
    exit()

def process_test(test_ids, train):
    test_ids['present_in_train'] = test_ids['user_id'].isin(train['user_id'])
    print(test_ids[test_ids['present_in_train'] == True])
    test_ids['recommendations'] = np.where((test_ids['present_in_train'] == True), get_user_cluster(), get_decision_tree())