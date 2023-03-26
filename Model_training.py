import numpy as np
import pandas as pd
import pickle
from neural_network import Model
from neural_network import forward_propagation


columns = [str(x) for x in range(401)]

df = pd.read_csv('img_data_2.csv',names=columns)
array_set = np.array(df)
np.random.shuffle(array_set)
division_number = 4000
X_train, Y_train = array_set[:division_number,:-1], array_set[:division_number,-1].reshape(-1,1).astype(int)
X_test, Y_test = array_set[division_number:,:-1], array_set[division_number:,-1].reshape(-1,1).astype(int)

hidden_n_nodes = (40,)
K = 10
l = 1
alpha = 1.2
n_iterations = 400

costs, theta_sets = Model(X_train,Y_train,hidden_n_nodes,K,l,alpha,n_iterations)
sigmoid_sets, a_sets = forward_propagation(X_test,theta_sets)
predicted_y = np.argmax(a_sets[-1],axis=1).reshape(-1,1)
accuracy = sum(((predicted_y==Y_test).astype(int)))/len(Y_test)*100
print(np.hstack((predicted_y,Y_test))[:20])
print(accuracy)
#print(np.hstack((predicted_y,Y_test))[:10])

with open('theta_set.txt','wb') as f:
    pickle.dump(theta_sets,f)

