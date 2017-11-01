import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

## A measure of error: mean square error
def calculate_MSE(theta,data):
    total_square_error = 0.0
    for i in range(len(dataset)):
        total_square_error += (theta[0]+theta[1]*data[i,0]-data[i,1])**2
    return total_square_error/len(dataset)

## Not useful in the process, just for the sake of completeness
def cost_function():
    total_square_error = 0.0
    for i in range(len(dataset)):
        total_square_error += (theta[0]+theta[1]*data[i,0]-data[i,1])**2
    return total_square_error/2

def theta_monitor(theta,j):
    if(j%100==0):
        print("theta_0: {0:f}, theta_1: {1:f}".format(theta[0],theta[1]))

## Use gradient decent to minimize the error
def batch_gradient_decent(theta,data,learning_rate=1e-5,number_of_steps=5000,threshold=None):
    for j in range(number_of_steps):
        partial_derivative_sum = [0.0,0.0]
        theta_error=[0.0,0.0]
        for i in range(len(dataset)):
            partial_derivative_sum[0] += data[i,1]-theta[0]-theta[1]*data[i,0]
            partial_derivative_sum[1] += (data[i,1]-theta[0]-theta[1]*data[i,0])*data[i,0]
        theta_error[0] = learning_rate*partial_derivative_sum[0]
        theta_error[1] = learning_rate*partial_derivative_sum[1]
        theta[0] += theta_error[0]
        theta[1] += theta_error[1]
        theta_monitor(theta,j)

        if((threshold is not None) and (theta_error[0]<threshold) and (theta_error[1]<threshold)):
            return theta
    return theta

## Stochastic gradient decent converges faster
def stochastic_gradient_decent(theta,data,learning_rate=1e-5,number_of_steps=5000,threshold=None):
    for j in range(number_of_steps):
        partial_derivative = [0.0,0.0]
        theta_old=theta
        for i in range(len(dataset)):
            partial_derivative[0] = data[i,1]-theta[0]-theta[1]*data[i,0]
            partial_derivative[1] = (data[i,1]-theta[0]-theta[1]*data[i,0])*data[i,0]

            theta[0] += learning_rate*partial_derivative[0]
            theta[1] += learning_rate*partial_derivative[1]
        ## This part of threshold should be improved
        if ((threshold is not None) and (abs(theta[0]-theta_old[0]) < threshold) and (abs(theta[1]-theta_old[1]) < threshold)):
            return theta
        theta_monitor(theta, j)
    return theta

# def plot():
#     fig =

if __name__ == '__main__':
    dataset_whole = pd.read_csv('../../dataset/LOL/stats1.csv',index_col=False,usecols=['goldearned','goldspent'])
    ## Normalize the data
    # dataset = dataset_whole.head(5000)/10000
    dataset = dataset_whole/10000
    del dataset_whole
    data = dataset.as_matrix()

    ## If learning rate is bigger than some value, the variables will diverge
    ## threshold is learning_rate * 1e2

    # theta = [0.0, 0.0]
    # print("theta_0:{0:f}  theta_1:{1:f}".format(theta[0], theta[1]))
    # print("MSE before: {:f}\n".format(calculate_MSE(theta,data)))
    #
    # start_time = time.time()
    # theta = batch_gradient_decent(theta,data,learning_rate=1e-7,threshold=-5)
    # print("Batch gradient decent use time: {}".format(time.time()-start_time))
    # print("theta_0:{0:f}  theta_1:{1:f}".format(theta[0], theta[1]))
    # print("MSE after: {:f}\n".format(calculate_MSE(theta,data)))

    theta = [0.0, 0.0]
    print("theta_0:{0:f}  theta_1:{1:f}".format(theta[0], theta[1]))
    start_time = time.time()
    theta = stochastic_gradient_decent(theta,data,learning_rate=1e-4,threshold=1e-2)
    print("Stochastic gradient decent use time: {}".format(time.time()-start_time))
    print("theta_0:{0:f}  theta_1:{1:f}".format(theta[0], theta[1]))
    print("MSE after: {:f}\n".format(calculate_MSE(theta,data)))