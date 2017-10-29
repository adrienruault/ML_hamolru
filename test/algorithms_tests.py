import numpy as np

import sys 
import os
sys.path.append(os.path.relpath("../src"))
from algorithms import *

from utilities import *



def main():
    test_ls_grid_GD_SGD()
    test_newton_GD_reg_logistic()
    




def test_ls_grid_GD_SGD():
    print()
    print("BEGINNING OF TEST_LS_GRID_GD_SGD")
    height, weight, gender = load_data_from_ex02(sub_sample=False, add_outlier=False)
    x, mean_x, std_x = standardize(height)
    y, tx = build_model_data(x, weight)
    
    ls_w, ls_loss = least_squares(y, tx)
    
    w0_grid_test = np.linspace(-100, 100, 100)
    w1_grid_test = np.linspace(-100, 100, 100)
    grid_w, grid_loss = grid_search(y, tx, w0_grid_test, w1_grid_test)
    
    initial_w = np.array([0, 0])
    gamma_GD = 0.7
    gamma_GD_mae = 10
    max_iters = 500
    GD_w, GD_loss = gradient_descent(y, tx, initial_w, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_div=10)
    GD_w_mae, GD_loss_mae = gradient_descent(y, tx, initial_w, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_div=10)

    gamma_SGD = 0.1
    gamma_SGD_mae = 2
    max_iters = 100
    batch_size = 200
    SGD_w, SGD_loss = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_div=10)
    SGD_w_mae, SGD_loss_mae = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_div=10)
    
    print("Weights summary:")
    print("ls_w:", ls_w)
    print("grid_w:", grid_w)
    print("GD_w:", GD_w)
    print("GD_w_mae:", GD_w_mae)
    print("SGD_w:", SGD_w)
    print("SGD_w_mae", SGD_w_mae)
    
    print("END OF TEST_LS_GRID_GD_SGD")
    print()
    
    return 0;



    
    
def test_newton_GD_reg_logistic():
    
    print()
    print("BEGINNING OF TEST_NEWTON_GD_REG_LOGISTIC")
    
    
    # load data.
    height, weight, gender = load_data_from_ex02()

    # build sampled x and y.
    seed = 1
    y = np.expand_dims(gender, axis=1)
    X = np.c_[height.reshape(-1), weight.reshape(-1)]
    y, X = sample_data(y, X, seed, size_samples=200)
    x, mean_x, std_x = standardize(X)
    
    max_iter = 10000
    gamma_gd = 0.01
    gamma_newt = 4.5
    lambda_ = 0.1
    threshold = 1e-8
    thresh_test_div = 100

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    
    initial_w = np.zeros((tx.shape[1], 1))
    
    loss, w = gradient_descent(y, tx, initial_w, max_iter, gamma_gd, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_div=thresh_test_div, update_gamma=False)
    
    loss, w = newton(y, tx, initial_w, max_iter, gamma_newt, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_div=thresh_test_div, update_gamma=False)
    
    
    print("END OF TEST_NEWTON_GD_REG_LOGISTIC")
    print()
    
    
    return 0




if (__name__ == "__main__"):
    main()



