import proj1_helpers as utils
import numpy as np
import algorithms as ML_alg
import preprocessing_functions as prf
from joblib import Parallel, delayed


def main():
    (y, x, event_ids) = utils.load_csv_data("../data/train.csv")

    x_nan_to_mean = prf.put_nan_to_mean(x, y)
    y_bin = prf.pass_data_to_zero_one(y).reshape([y.shape[0],1])


    std_x = prf.standardize(x_nan_to_mean)



    degree_min = 30 * [1]
    degree_max = 30 * [1]
    
    Parallel(n_jobs = 2)(delayed(ML_alg.tuner_degree_lambda)(y = y_bin, x = std_x,\
                    degree_min = degree_min, degree_max = degree_max, lambda_min = -4,\
                    lambda_max = 0, nb_lambda = 30, k_fold=4, seed=1,\
                    max_iters=10000, gamma=gam, cost="reg_logistic", tol=1e-3,\
                    thresh_test_div=10, update_gamma=False) for gam in [1e-3,1e-4])

    
    best_degree, best_lambda, min_rmse_te, min_rmse_tr = ML_alg.tuner_degree_lambda(y=y_bin, x=std_x,\
                    degree_min = degree_min, degree_max = degree_max, lambda_min = -4,\
                    lambda_max = 0, nb_lambda = 30, k_fold=4, seed=1,\
                    max_iters=10000, gamma=0.00001, cost="reg_logistic", tol=1e-3,\
                    thresh_test_div=10, update_gamma=False)
    
    result_file  = open("results_{deg}.txt".format(deg = 1), 'w')
    result_file.close()
    
    
    
    
if (__name__ == "__main__"):
    main()
