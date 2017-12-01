import pandas as pd
import numpy as np
import math
import multiprocessing as mp

def load_data():
    dataset = pd.read_csv('../../supervised_learning/dataset/Iris/bezdekIris.data',
                          header=None, index_col=False,
                          names=['sepal_len','sepal_wid','petal_len','petal_wid','class'])
    return dataset
def vector_distance(x,y):
    if len(x) != len(y):
        raise ValueError("Can't compute vector distance when the length of vectors are not the same")
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return math.sqrt(sum)

def list_equal(l1,l2):
    if len(l1) != len(l2):
        raise ValueError("Can't compare list when the length of two lists are different.")
    if all(l1[i] == l2[i] for i in range(len(l1))):
        return True
    return False

def find_centroids(data_matrix, centroid_dict, q):
    # List of centroids for the dataset
    centroids = ['c0' for _ in range(150)]

    while True:
        c0_list = []
        c1_list = []
        c2_list = []
        # E step, determine centroids for each data point
        for i in range(len(data_matrix)):
            min_distance = np.argmin([
                vector_distance(data_matrix[i], centroid_dict['c0']),
                vector_distance(data_matrix[i], centroid_dict['c1']),
                vector_distance(data_matrix[i], centroid_dict['c2'])
            ])
            if min_distance == 0:
                centroids[i] = 'c0'
                c0_list.append(data_matrix[i])
            if min_distance == 1:
                centroids[i] = 'c1'
                c1_list.append(data_matrix[i])
            if min_distance == 2:
                centroids[i] = 'c2'
                c2_list.append(data_matrix[i])

        # M step, find the new centroid location
        c0_new = np.mean(c0_list, axis=0)
        c1_new = np.mean(c1_list, axis=0)
        c2_new = np.mean(c2_list, axis=0)

        # break loop if the centroids do not change
        if list_equal(centroid_dict['c0'],c0_new) and list_equal(centroid_dict['c1'], c1_new) and list_equal(centroid_dict['c2'], c2_new):
            break
        else:
            centroid_dict['c0'] = c0_new
            centroid_dict['c1'] = c1_new
            centroid_dict['c2'] = c2_new
    q.put(centroids)

def classes_count(centroids):
    # cluster indice according to the centroids
    c0_index_list = []
    c1_index_list = []
    c2_index_list = []
    for i in range(len(centroids)):
        if centroids[i] == 'c0': c0_index_list.append(i)
        if centroids[i] == 'c1': c1_index_list.append(i)
        if centroids[i] == 'c2': c2_index_list.append(i)

    # Count classes in different clusters (which have diferent centroids)
    c0_dict = {'Setosa':0, 'Versicolour':0, 'Virginica':0}
    c1_dict = {'Setosa':0, 'Versicolour':0, 'Virginica':0}
    c2_dict = {'Setosa':0, 'Versicolour':0, 'Virginica':0}
    for j in range(len(c0_index_list)):
        i = c0_index_list[j]
        if dataset.iloc[i, -1] == 'Iris-setosa': c0_dict['Setosa'] += 1
        if dataset.iloc[i, -1] == 'Iris-versicolor': c0_dict['Versicolour'] += 1
        if dataset.iloc[i, -1] == 'Iris-virginica': c0_dict['Virginica'] += 1
    for j in range(len(c1_index_list)):
        i = c1_index_list[j]
        if dataset.iloc[i, -1] == 'Iris-setosa': c1_dict['Setosa'] += 1
        if dataset.iloc[i, -1] == 'Iris-versicolor': c1_dict['Versicolour'] += 1
        if dataset.iloc[i, -1] == 'Iris-virginica': c1_dict['Virginica'] += 1
    for j in range(len(c2_index_list)):
        i = c2_index_list[j]
        if dataset.iloc[i, -1] == 'Iris-setosa': c2_dict['Setosa'] += 1
        if dataset.iloc[i, -1] == 'Iris-versicolor': c2_dict['Versicolour'] += 1
        if dataset.iloc[i, -1] == 'Iris-virginica': c2_dict['Virginica'] += 1

    print(c0_dict)
    print(c1_dict)
    print(c2_dict)
    print('')



# Main function
if __name__ == '__main__':
    dataset = load_data()
    data_matrix = dataset.as_matrix(['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])

    # Initialize the centroids
    initial_centroids_0 = {'c0': data_matrix[0], 'c1': data_matrix[50], 'c2': data_matrix[100]}
    initial_centroids_1 = {'c0': data_matrix[8], 'c1': data_matrix[51], 'c2': data_matrix[101]}
    initial_centroids_2 = {'c0': data_matrix[34], 'c1': data_matrix[78], 'c2': data_matrix[105]}
    initial_centroids_3 = {'c0': data_matrix[48], 'c1': data_matrix[35], 'c2': data_matrix[70]}

    # Training step: cluster instances and find centroid for each instances
    q0 = mp.Queue()
    q1 = mp.Queue()
    q2 = mp.Queue()
    q3 = mp.Queue()

    p0 = mp.Process(target=find_centroids, args=(data_matrix, initial_centroids_0, q0,))
    p1 = mp.Process(target=find_centroids, args=(data_matrix, initial_centroids_1, q1,))
    p2 = mp.Process(target=find_centroids, args=(data_matrix, initial_centroids_2, q2,))
    p3 = mp.Process(target=find_centroids, args=(data_matrix, initial_centroids_3, q3,))

    p0.start()
    p1.start()
    p2.start()
    p3.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()

    centroids_0 = q0.get()
    centroids_1 = q1.get()
    centroids_2 = q2.get()
    centroids_3 = q3.get()

    # Count number of instances in different classes
    classes_count(centroids_0)
    classes_count(centroids_1)
    classes_count(centroids_2)
    classes_count(centroids_3)



