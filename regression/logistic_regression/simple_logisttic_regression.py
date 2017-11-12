import numpy as np
import pandas as pd

def load_data():
    dataset = pd.read_csv('../../dataset/LOL/stats1.csv',index_col=False,
                          usecols=[
                              'win','deaths','assists','longesttimespentliving','totdmgdealt',
                              'totheal','totdmgtaken','goldearned','totcctimedealt','champlvl'
                          ])
    return dataset

## A measure of precision
def calculate_precision(data,theta):
    x = data[:,1:]
    x = np.insert(x,0,1,axis=1)
    y = data[:,0]
    res = x.dot(theta)
    match_count = 0
    for i in range(len(y)):
        if res[i] >= 0 and y[i] == 1: match_count += 1
        if res[i] < 0 and y[i] == 0: match_count += 1
    return match_count/len(y)


def sigmoid_function(z):
    return 1/(1+np.exp(-z))

# To see how theta value changes
def theta_monitor(theta,j):
    if(j%5==0):
        print_theta(theta)

# Stochastic gradient decent process without regulation
def stochastic_gradient_decent(theta,data,learning_rate=1e-5,number_of_steps=500,threshold=None):
    for j in range(number_of_steps):
        partial_derivative = np.zeros(len(theta))
        theta_old = theta

        for i in range(len(data)):
            row = data[i]
            x = row[1:]
            x = np.insert(x,0,1)
            y = row[0]
            partial_derivative = (y-sigmoid_function(theta.dot(x)))*x
            theta += learning_rate*partial_derivative
        if threshold is not None and all(theta[k] < threshold for k in range(len(theta))):
            return theta
        theta_monitor(theta, j)
    return theta

def print_theta(theta):
    formated_list = ['{:>4}' for t in theta]
    s = ','.join(formated_list)
    print(s.format(*theta))

if __name__ == '__main__':
    dataset = load_data()
    print('load data success')
    dataset.loc[:,'longesttimespentliving'] /= 1000
    dataset.loc[:, 'totdmgdealt'] /= 100000
    dataset.loc[:, 'totheal'] /= 10000
    dataset.loc[:, 'totdmgtaken'] /= 10000
    dataset.loc[:, 'goldearned'] /= 10000
    dataset.loc[:, 'totcctimedealt'] /= 100
    dataset.loc[:, 'champlvl'] /= 10
    data = dataset.as_matrix()

    theta = np.zeros(len(data[0]))
    print('theta start value:')
    print_theta(theta)
    # Training step
    theta = stochastic_gradient_decent(theta, data, learning_rate=1e-4, number_of_steps=80, threshold=1e-2)
    # Print out the result
    print('theta after training:')
    print_theta(theta)
    print('Precision: {}'.format(calculate_precision(data,theta)))
