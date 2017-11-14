%load_ext autoreload
%autoreload 2
import sys 
import os

import multiprocessing as mp

import algorithms as alg
import numpy as np
import proj1_helpers as utils
import preprocessing_functions as prf




def main():
    parallel_tuner()




def deg_min_max_generator_parallel(deg_min = 0, deg_max = 10):
    inputs = np.ones((30, 2, 30))
    for i in range(30):
        inputs[i, 0, i] = deg_min
        inputs[i, 1, i] = deg_max
        
    return inputs.astype(int).tolist()






# define a example function
def target_func(inputs_min_max_deg, output):
    degree_min = inputs_min_max_deg[0]
    degree_max = inputs_min_max_deg[1]
    seeds = range(25)
    
    (y, x, event_ids) = utils.load_csv_data("../Data/train.csv")
    x_nan_to_mean = prf.put_nan_to_mean(x, y)
    y_bin = prf.pass_data_to_zero_one(y).reshape([y.shape[0],1])
    std_x = prf.standardize(x_nan_to_mean)
    
    answer = alg.tuner_degree_lambda(y = y_bin, x = std_x, degree_min = degree_min, degree_max = degree_max,\
                                        lambda_min = 0, lambda_max = 0, nb_lambda = 1, k_fold = 4, seeds = seeds,\
                                        max_iters = 100000, gamma = 1e-5, cost = "reg_logistic", tol = 1e-4,\
                                        thresh_test_div=10, update_gamma=False)
    
    output.put(answer)
    
    
    
    
    
    
    
    
def parallel_tuner():
    random.seed(123)

    # Define an output queue
    output = mp.Queue()

    inputs_min_max_deg = deg_min_max_generator_parallel(deg_min = 0, deg_max = 10)
    
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=target_func, args=(input_elem, output)) for input_elem in inputs_min_max_deg]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]
    
    for i in range(30):
        result_i = results[i]
        file_i = open("tuning_dim_{i}.txt".format(i=i), 'w')
        file_i.write("Best degree:", result_i[0][0], "with index:", result_i[0][1])
        file_i.write("Best lambda:", result_i[1][0], "with index:", result_i[1][1])
        file_i.write("\nmean_rmse_te:")
        file_i.write(result_i[2][0])
        file_i.write("std_rmse_te:")
        file_i.write(result_i[2][1])
        file_i.write("\nmean_rmse_tr:")
        file_i.write(result_i[3][0])
        file_i.write("std_rmse_tr:")
        file_i.write(result_i[3][1])
        file_i.close()
    
    
    

if (__name__ == "__main__"):
    main()
    
    
    
    



