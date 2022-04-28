import os
import numpy as np
import time
import logging
import sys
import math
import subprocess, shlex
from shutil import copyfile
import json
from threading import Timer
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def build_model(regression_mod='RF', test_mode=True, training_X=[], training_Y=[]):
    """
    to build the specified regression model, given the training data
    :param regression_mod: the regression model to build
    :param test_mode: won't tune the hyper-parameters if test_mode == False
    :param training_X: the array of training features
    :param training_Y: the array of training label
    :return: the trained model
    """
    model = None
    if regression_mod == 'RF':
        model = RandomForestRegressor(random_state=0)
        max = 3
        if len(training_X)>30: # enlarge the hyperparameter range if samples are enough
            max = 6
        param = {'n_estimators': np.arange(10, 100, 20),
                 'criterion': ('mse', 'mae'),
                 'max_features': ('auto', 'sqrt', 'log2'),
                 'min_samples_split': np.arange(2, max, 1)
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = RandomForestRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'KNN':
        min = 2
        max = 3
        if len(training_X)>30:
            max = 16
            min = 5
        model = KNeighborsRegressor(n_neighbors=min)
        param = {'n_neighbors': np.arange(2, max, 2),
                 'weights': ('uniform', 'distance'),
                 'algorithm': ['auto'],  # 'ball_tree','kd_tree'),
                 'leaf_size': [10, 30, 50, 70, 90],
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KNeighborsRegressor(**gridS.best_params_)

    elif regression_mod == 'SVR':
        model = SVR()
        param = {'kernel': ('linear', 'rbf'),
                 'degree': [2, 3, 4, 5],
                 'gamma': ('scale', 'auto'),
                 'coef0': [0, 2, 4, 6, 8, 10],
                 'epsilon': [0.01, 1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = SVR(**gridS.best_params_)

    elif regression_mod == 'DT':
        model = DecisionTreeRegressor(random_state=0)
        max = 3
        if len(training_X)>30:
            max = 6
        param = {'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'min_samples_split': np.arange(2, max, 1)
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = DecisionTreeRegressor(**gridS.best_params_, random_state=0)

    elif regression_mod == 'LR':
        model = LinearRegression()
        param = {'fit_intercept': ('True', 'False'),
                 # 'normalize': ('True', 'False'),
                 'n_jobs': [1, -1]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = LinearRegression(**gridS.best_params_)

    elif regression_mod == 'KR':
        x1 = np.arange(0.1, 5, 0.5)
        model = KernelRidge()
        param = {'alpha': x1,
                 'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                 'coef0': [1, 2, 3, 4, 5]
                 }
        if not test_mode:
            # print('Hyperparameter Tuning...')
            gridS = GridSearchCV(model, param)
            gridS.fit(training_X, training_Y)
            model = KernelRidge(**gridS.best_params_)

    return model


# def minibatches(data_generator, minibatch_size):
#     """
#     Args:
#         data_generator: generator of (img, formulas) tuples
#         minibatch_size: (int)
#
#     Returns:
#         list of tuples
#
#     """
#     x_batch, y_batch = [], []
#     for (x, y) in data_generator:
#         if len(x_batch) == minibatch_size:
#             yield x_batch, y_batch
#             x_batch, y_batch = [], []
#
#         x_batch += [x]
#         y_batch += [y]
#
#     if len(x_batch) != 0:
#         yield x_batch, y_batch
#

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# def run(cmd, timeout_sec):
#     """Run cmd in the shell with timeout"""
#     proc = subprocess.Popen(cmd, shell=True)
#     kill_proc = lambda p: p.kill()
#     timer = Timer(timeout_sec, kill_proc, [proc])
#     try:
#         timer.start()
#         stdout,stderr = proc.communicate()
#     finally:
#         timer.cancel()


def init_dir(dir_name):
    """Creates directory if it does not exists"""
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


# def init_file(path_file, mode="a"):
#     """Makes sure that a given file exists"""
#     with open(path_file, mode) as f:
#         pass
#
#
# def get_files(dir_name):
#     """ Get files in a directory """
#     files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
#     return files
#
#
# def delete_file(path_file):
#     try:
#         os.remove(path_file)
#     except Exception:
#         pass

