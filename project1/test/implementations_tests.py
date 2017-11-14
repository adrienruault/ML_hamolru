
import numpy as np

import sys 
import os
sys.path.append(os.path.relpath("../src"))
from algorithms import grid_search, gradient_descent, stochastic_gradient_descent
from implementations import *

from utilities import *




def main():
    testing_ls_ridge()
    testing_logistic()


def testing_ls_ridge():
    print()
    print("BEGINNING OF TESTING_LS_RIDGE")
    height, weight, gender = load_data_from_ex02(sub_sample=False, add_outlier=False)
    x, mean_x, std_x = standardize(height)
    y, tx = build_model_data(x, weight)
        
    w0_grid_test = np.linspace(-100, 100, 100)
    w1_grid_test = np.linspace(-100, 100, 100)
    grid_w, grid_loss = grid_search(y, tx, w0_grid_test, w1_grid_test)
    
    initial_w = [0, 0]
    gamma_GD = 0.7
    gamma_GD_mae = 10
    max_iters = 500
    GD_w, GD_loss = gradient_descent(y, tx, initial_w, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_div=10)
    GD_w_mae, GD_loss_mae = gradient_descent(y, tx, initial_w, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_div=10)

    gamma_SGD = 0.01
    gamma_SGD_mae = 0.1
    max_iters = 1000
    batch_size = 1
    SGD_w, SGD_loss = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_SGD, cost='mse', tol=1e-4, thresh_test_div=100)
    SGD_w_mae, SGD_loss_mae = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_SGD_mae, cost='mae', tol=1e-2, thresh_test_div=100)
    
    
    
    gamma_ridge = 0.01
    lambda_ridge = 0.3
    max_iters = 10000
    
    ridge_w, ridge_loss = gradient_descent(y, tx, initial_w, max_iters, gamma_ridge, cost='ridge', lambda_ = lambda_ridge, tol=1e-8, thresh_test_div=10)
    
    
    
    max_iters_test = 1000
    gamma_test = 0.01
    test_w_GD, test_loss_GD = least_squares_GD(y, tx, initial_w, max_iters_test, gamma_test)
    test_w_SGD, test_loss_SGD = least_squares_SGD(y, tx, initial_w, max_iters_test, gamma_test)
    test_w_ls, test_loss_ls = least_squares(y, tx)
    test_w_ridge, test_loss_ridge = ridge_regression(y, tx, lambda_ridge)

    print("Weights summary:")
    print("grid_w:", grid_w)
    print("GD_w:", GD_w)
    print("GD_w_mae:", GD_w_mae)
    print("SGD_w:", SGD_w)
    print("SGD_w_mae", SGD_w_mae)
    print("ridge_w:", ridge_w)
    print("test_w_GD:", test_w_GD)
    print("test_w_SGD:", test_w_SGD)
    print("test_w_ls:", test_w_ls)
    print("test_w_ridge:", test_w_ridge)

    

    print("END OF TESTING_LS_RIDGE")
    print()
    
    
    return 0;












def testing_logistic():
    print()
    print("BEGINNING OF TESTING_LOGISTIC")
    # load data.
    height, weight, gender = load_data_from_ex02()

    # build sampled x and y.
    seed = 1
    y = np.expand_dims(gender, axis=1)
    X = np.c_[height.reshape(-1), weight.reshape(-1)]
    y, X = sample_data(y, X, seed, size_samples=200)
    x, mean_x, std_x = standardize(X)
    
    max_iters = 100000
    gamma = 0.001

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]

    initial_w = np.zeros((tx.shape[1], 1))
    
    lambda_ = 0.3
    
    w_logistic, loss_logistic = gradient_descent(y, tx, initial_w, max_iters, gamma, cost='logistic',\
                                                 lambda_=0, tol=1e-15, thresh_test_div=10, update_gamma=False)
    
    w_test_log, loss_test_log = logistic_regression(y, tx, initial_w, max_iters, gamma)
    
    w_reg_log, loss_reg_log = gradient_descent(y, tx, initial_w, max_iters, gamma, cost='reg_logistic',\
                                                 lambda_=lambda_, tol=1e-15, thresh_test_div=10, update_gamma=False)
    
    w_test_reg_log, loss_test_reg_log = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
    
    print("Weights summary:")
    print("w_logistic:", w_logistic[:,0])
    print("w_test_log:", w_test_log[:,0])

    print("END OF TESTING_LOGISTIC")
    print()
    
    return 0


if (__name__ == "__main__"):
    main()




