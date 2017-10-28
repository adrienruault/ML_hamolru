import proj1_helpers as utils
import numpy as np
import matplotlib.pyplot as plt
import algorithms as ML_alg
import preprocessing_functions as prf


def evaluate(tx,y,w):
    true = 0
    false = 0
    for i in range(y.shape[0]):
        res = tx[i].dot(w.reshape([w.shape[0],1]))
        if(abs(1-res) < abs(res)):
            if (y[i] == 1):
                true += 1
            else:
                false += 1
        if(abs(1-res) > abs(res)):
            if (y[i] == 0):
                true += 1
            else:
                false += 1
    return (true,false)

def main():
    (y, tx, event_ids) = utils.load_csv_data("../Data/train.csv")

    std_tx = ML_alg.build_poly(prf.standardize(prf.put_nan_to_mean(tx,y)),degree=3)

    y_bin = prf.pass_data_to_zero_one(y)

    initial_w = np.zeros([std_tx.shape[1],1])

    (w, loss ) = ML_alg.gradient_descent_without_loss(tx = std_tx, y = y_bin, initial_w = initial_w,cost = 'reg_logistic',
                          lambda_= 0.5, gamma = 1e-7, update_gamma= False, max_iters = 10000)#, batch_size = 1000)


    (true, false) = evaluate(std_tx, y_bin, w)

    print(true / (true + false))
    
if(__name__ == "__main__"):
    main()