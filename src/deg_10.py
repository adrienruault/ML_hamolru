
import numpy as np
import algorithms as alg
import proj1_helpers as utils


def main():
    degree = [10]*30
    (y, x, event_ids) = utils.load_csv_data("../Data/train.csv")

    x_pred = data_nan_estimated_by_regression(x, y)

    tx = alg.build_poly(x_pred, degree)
    y = y.reshape([y.shape[0],1])

    w, loss = alg.ridge_regression(y, tx, 0.00610540229659)
    rep = utils.predict_labels(w, tx)

    #rep = utils.evaluate(tx, y_bin, w)
    rep = rep.squeeze()
    y = y.squeeze()
    print(np.abs(rep + y)/2)
    
    
    
    
    