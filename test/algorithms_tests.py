import numpy as np

import sys 
import os
sys.path.append(os.path.relpath("../src"))
from algorithms import *

from utilities import *






def train_test_split_demo(x, y, degree, ratio, seed):
    """polynomial regression with different split ratios and different degrees."""
    train_set, val_set = split_data(x, y, ratio, seed)
    
    x_train = train_set[0]
    x_val = val_set[0]
    
    y_train = train_set[1]
    y_val = val_set[1]
    
    tx_train = build_poly(x_train, degree)
    tx_val = build_poly(x_val, degree)
        
    loss, w = least_squares(y_train, tx_train)
    
    train_loss = compute_loss(y_train, tx_train, w, cost = "mse")
    val_loss = compute_loss(y_val, tx_val, w, cost = "mse")
    
    rmse_tr = np.sqrt(2 * train_loss)
    rmse_te = np.sqrt(2 * val_loss)

    print("proportion={p}, degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
          p=ratio, d=degree, tr=rmse_tr, te=rmse_te))


    







def test_ls_grid_GD_SGD():
    height, weight, gender = load_data_from_ex02(sub_sample=False, add_outlier=False)
    x, mean_x, std_x = standardize(height)
    y, tx = build_model_data(x, weight)
    
    ls_loss, ls_w = least_squares(y, tx)
    
    w0_grid_test = np.linspace(-100, 100, 100)
    w1_grid_test = np.linspace(-100, 100, 100)
    grid_loss, grid_w = grid_search(y, tx, w0_grid_test, w1_grid_test)
    
    initial_w = np.array([0, 0])
    gamma_GD = 0.7
    gamma_GD_mae = 10
    max_iters = 500
    GD_loss, GD_w = gradient_descent(y, tx, initial_w, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_conv=10)
    GD_loss_mae, GD_w_mae = gradient_descent(y, tx, initial_w, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_conv=10)

    gamma_SGD = 0.1
    gamma_SGD_mae = 2
    max_iters = 100
    batch_size = 200
    SGD_loss, SGD_w = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD, cost='mse', tol=1e-2, thresh_test_conv=10)
    SGD_loss_mae, SGD_w_mae = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma_GD_mae, cost='mae', tol=1e-2, thresh_test_conv=10)
    
    
    print("ls_w:", ls_w)
    print("grid_w:", grid_w)
    print("GD_w:", GD_w[len(GD_w)-1])
    print("GD_w_mae:", GD_w_mae[len(GD_w_mae)-1])
    print("SGD_w:", SGD_w[len(GD_w_mae)-1])
    print("SGD_w_mae", SGD_w_mae[len(GD_w_mae)-1])
    
    return 0;



    
    
def test_newton_gradient_descent_reg_logistic():
    # load data.
    height, weight, gender = load_data_from_ex02()

    # build sampled x and y.
    seed = 1
    y = np.expand_dims(gender, axis=1)
    X = np.c_[height.reshape(-1), weight.reshape(-1)]
    y, X = sample_data(y, X, seed, size_samples=200)
    x, mean_x, std_x = standardize(X)
    
    print(y.shape)
    print(x.shape)
    
    max_iter = 10000
    gamma_gd = 0.01
    gamma_newt = 4.5
    lambda_ = 0.1
    threshold = 1e-8

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    
    initial_w = np.zeros((tx.shape[1], 1))
    
    loss, w = gradient_descent(y, tx, initial_w, max_iter, gamma_gd, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_conv=10, update_gamma=False)
    
    loss, w = newton(y, tx, initial_w, max_iter, gamma_newt, cost='reg_logistic', lambda_=lambda_, tol=threshold, thresh_test_conv=10, update_gamma=False)
    
    return 0




