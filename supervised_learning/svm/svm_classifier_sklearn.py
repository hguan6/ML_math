"""
Due to the complexity of numerical computation of SVM, sklearn is used as the learning library,

Data available from https://www.kaggle.com/paololol/league-of-legends-ranked-matches/data
"""

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit

def load_data():
    dataset = pd.read_csv('../dataset/LOL/stats1.csv',index_col=False,
                          usecols=[
                              'win', 'kills','deaths','assists','longesttimespentliving','totdmgdealt',
                              'totheal','totdmgtaken','goldearned','totcctimedealt','champlvl'
                          ])
    return dataset

# Seperate the dataset into training set, cross validation set, and test set
def preprocessing(dataset):
    dataset.loc[:, 'kills'] /= 10
    dataset.loc[:, 'deaths'] /= 10
    dataset.loc[:, 'assists'] /= 10
    dataset.loc[:, 'longesttimespentliving'] /= 1000
    dataset.loc[:, 'totdmgdealt'] /= 100000
    dataset.loc[:, 'totheal'] /= 10000
    dataset.loc[:, 'totdmgtaken'] /= 10000
    dataset.loc[:, 'goldearned'] /= 10000
    dataset.loc[:, 'totcctimedealt'] /= 1000
    dataset.loc[:, 'champlvl'] /= 10
    y = dataset['win'].as_matrix()
    X = dataset.drop('win', axis=1).as_matrix()
    sss_test = StratifiedShuffleSplit(n_splits=3, test_size=0.2,random_state=0)
    sss_cv = StratifiedShuffleSplit(n_splits=3, test_size=0.25,random_state=0)
    for train_cv_index, test_index in sss_test.split(X,y):
        X_train_cv, X_test = X[train_cv_index], X[test_index]
        y_train_cv, y_test = y[train_cv_index], y[test_index]
    for train_index, cv_index in sss_cv.split(X_train_cv,y_train_cv):
        X_train, X_cv = X[train_index], X[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
    return X_train,X_cv,X_test,y_train,y_cv,y_test


def train(X_train, X_cv, y_train, y_cv, C, q, fit_intercept=True, intercept_scaling=1):
    clf = LinearSVC(C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train,y_train)
    cv_score = clf.score(X_cv,y_cv)
    q.put(clf)
    q.put(train_score)
    q.put(cv_score)

def train_and_test(X_train, X_test, y_train, y_test, C=1.0, fit_intercept=True, intercept_scaling=1):
    clf = LinearSVC(C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
    clf.fit(X_train,y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    return train_score, test_score, clf.coef_, clf.intercept_

if __name__ == '__main__':
    dataset = load_data()
    X_train, X_cv, X_test, y_train, y_cv, y_test = preprocessing(dataset)

    # q1 = mp.Queue()
    # q2 = mp.Queue()
    # q3 = mp.Queue()
    # q4 = mp.Queue()
    #
    # C1, C2, C3, C4 = 2.0, 1.0, 0.1, 0.05
    #
    # p1 = mp.Process(target=train, args=(X_train, X_cv, y_train, y_cv, C1, q1,))
    # p2 = mp.Process(target=train, args=(X_train, X_cv, y_train, y_cv, C2, q2,))
    # p3 = mp.Process(target=train, args=(X_train, X_cv, y_train, y_cv, C3, q3,))
    # p4 = mp.Process(target=train, args=(X_train, X_cv, y_train, y_cv, C4, q4,))
    #
    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    #
    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    #
    # clf_1 = q1.get()
    # clf_2 = q2.get()
    # clf_3 = q3.get()
    # clf_4 = q4.get()
    #
    # train_score_1 = q1.get()
    # train_score_2 = q2.get()
    # train_score_3 = q3.get()
    # train_score_4 = q4.get()
    #
    # cv_score_1 = q1.get()
    # cv_score_2 = q2.get()
    # cv_score_3 = q3.get()
    # cv_score_4 = q4.get()
    #
    # print('train score for C={} is: {}'.format(C1, train_score_1))
    # print('cv score for C={} is: {}'.format(C1, cv_score_1))
    # print('coef of clf_1: {}'.format(clf_1.coef_))
    # print('intercept of clf_1: {}\n'.format(clf_1.intercept_))
    #
    # print('train score for C={} is: {}'.format(C2, train_score_2))
    # print('cv score for C={} is: {}'.format(C2, cv_score_2))
    # print('coef of clf_2: {}'.format(clf_2.coef_))
    # print('intercept of clf_2: {}\n'.format(clf_2.intercept_))
    #
    # print('train score for C={} is: {}'.format(C3, train_score_3))
    # print('cv score for C={} is: {}'.format(C3, cv_score_3))
    # print('coef of clf_3: {}'.format(clf_3.coef_))
    # print('intercept of clf_3: {}\n'.format(clf_3.intercept_))
    #
    # print('train score for C={} is: {}'.format(C4, train_score_4))
    # print('cv score for C={} is: {}'.format(C4, cv_score_4))
    # print('coef of clf_4: {}'.format(clf_4.coef_))
    # print('intercept of clf_4: {}\n'.format(clf_4.intercept_))

    train_score, test_score, coef_, intercept_ = train_and_test(np.concatenate((X_train,X_cv),axis=0), X_test, np.concatenate((y_train,y_cv),axis=0), y_test, C=0.1)
    print('train score is {}, test score is {}'.format(train_score, test_score))
    print('coefficient is: {}, interception is: {}'.format(coef_, intercept_))