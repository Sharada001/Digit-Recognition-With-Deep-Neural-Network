import numpy as np

def sigmoid(z):
    return np.reciprocal(1+np.exp(-1*z))

def dataset_a(g_z):
    return np.hstack((np.ones((len(g_z),1)),g_z))

def dataset_y(Y,K):
    return np.fromfunction(np.vectorize(lambda i,j:1 if int(j)==Y[int(i),0] else 0),(len(Y),K))

def initial_theta_constructor(current_n_nodes,next_n_nodes):
    Init_Epsilon = 0.01
    return np.random.random((next_n_nodes,current_n_nodes))*2*Init_Epsilon - Init_Epsilon

def initial_theta_set(X,hidden_n_nodes,K):
    initial_theta_values = []
    all_n_nodes = (X.shape[1],*hidden_n_nodes,K)
    for i in range(len(all_n_nodes)-1):
        initial_theta_values.append(initial_theta_constructor(all_n_nodes[i]+1,all_n_nodes[i+1]))
    return initial_theta_values

def cost_function(y,a_sets,theta_sets,l):
    m = len(y)
    cost = (-1/m)*sum(sum(y*np.log(a_sets[-1])+(1-y)*np.log(1-a_sets[-1])))
    reg_err = 0
    for theta_set in theta_sets:
        reg_err += (l/(2*m))*sum(sum(np.power(theta_set[:,1:],2)))
    return cost + reg_err

def forward_propagation(X,theta_sets):
    sigmoid_sets = []
    a_sets = []
    a_sets.append(dataset_a(X))
    for theta_set in theta_sets:
        sigmoid_sets.append(sigmoid(a_sets[-1]@theta_set.T))
        a_sets.append(dataset_a(sigmoid_sets[-1]))
    a_sets[-1] = a_sets[-1][:,1:]
    return sigmoid_sets, a_sets

def backpropagation(sigmoid_sets,a_sets,theta_sets,l,y):
    m = len(y)
    reversed_delta = []
    reversed_delta.append(a_sets[-1]-y)
    for i in list(range(1,len(theta_sets)))[::-1]:
        reversed_delta.append((reversed_delta[-1]@theta_sets[i][:,1:])*(sigmoid_sets[i-1])*(1-(sigmoid_sets[i-1])))
    delta_sets = reversed_delta[::-1]
    gradient_sets = []
    for i in range(len(theta_sets)):
        gradient_sets.append((1/m)*(((a_sets[i]).T@delta_sets[i]).T + np.hstack((np.zeros((len(theta_sets[i]),1)),l*theta_sets[i][:,1:]))))
    return gradient_sets

def Model(X,Y,hidden_n_nodes,K,l,alpha,n_iterations):
    costs = [[],[]]
    y = dataset_y(Y,K)
    theta_sets = initial_theta_set(X,hidden_n_nodes,K)
    for x in range(n_iterations):
        print(x)
        sigmoid_sets, a_sets = forward_propagation(X,theta_sets)
        gradient_sets = backpropagation(sigmoid_sets,a_sets,theta_sets,l,y)
        for i in range(len(theta_sets)):
            theta_sets[i] -= alpha*gradient_sets[i]
        costs[0].append(x+1)
        costs[1].append(cost_function(y,a_sets,theta_sets,l))
    return costs, theta_sets




