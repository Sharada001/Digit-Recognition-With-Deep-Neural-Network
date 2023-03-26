import numpy as np
import pickle
from neural_network import forward_propagation

with open('theta_set.txt','rb') as f:
    theta_sets = pickle.load(f)

def predict_y_value(X_test):
    global theta_sets
    determiner = len(X_test.shape)
    if determiner==1:
        X_test = np.vstack((X_test,np.zeros(X_test.shape)))
    sigmoid_sets, a_sets = forward_propagation(X_test,theta_sets)
    max_value_indeces =  np.argmax(a_sets[-1],axis=1).reshape(-1,1)
    return max_value_indeces[0,0] if determiner==1 else max_value_indeces
