import preprocessing_functions as prf
import numpy as np
import algorithms as alg


def data_nan_estimated_by_regression(x, y):
    
    x_predicted = x
    x_predicted[x[:,:] == -999.0] = 0.
    print(x_predicted.shape)
    for i in range(30):
        x_clean, y_clean = prf.get_clean_data(x_predicted,y)
        x_train = np.c_[x_clean[:,:i],x_clean[:,i+1:]]
        x_i = x_clean[:,i]
        tx_train = alg.build_poly(x_train, 29*[1])
        w, loss = alg.least_squares(x_i, tx_train)

        # get rows with NaN on i's dimension
        x_nan_on_i = x[x[:,i] == 0.0]
        x_test = np.c_[x_nan_on_i[:,:i],x_nan_on_i[:,i+1:]]
        print(x_nan_on_i.shape)
        tx_test = alg.build_poly(x_test, 29*[1])

        y_pred = tx_test.dot(w)
        x_predicted[x_predicted[:,i] == 0.0, i] = y_pred
        
        
    # stndardisation of the matrix
    x_predicted = (x_predicted - np.mean(x_predicted, axis=0)) / np.std(x_predicted, axis=0)
    
    return x_predicted
    
    
    
