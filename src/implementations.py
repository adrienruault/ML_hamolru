

import numpy as np


#tested
def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    # number of training data
    N = tx.shape[0]
    
    # Define initial values of w and its associated mse loss
    w_k = initial_w
    loss_k = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w_k))**2
    
    # stopping criterion definition:
    n_iter = 0;
    while (n_iter < max_iters):
        # computation of the gradient
        grad = -(1/N) * np.transpose(tx).dot(y - tx.dot(w_k))
        
        # update w
        w_kp1 = w_k - gamma * grad
        
        # upsate loss wrt mse cost function
        loss_kp1 = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w_kp1))**2
        
        loss_k = loss_kp1
        w_k = w_kp1
        
        # update n_iter: number of iterations
        n_iter += 1
        
    # Printing the results
    try:
        initial_w.shape[1]
        print("least_squares_GD({bi}/{ti}): loss={l}, w = {w}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k[:,0]))
    except (IndexError, AttributeError):
        print("least_squares_GD({bi}/{ti}): loss={l}, w = {w}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k))
    return w_k, loss_k


















#tested
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # number of training data
    N = tx.shape[0]
    
    # Define initial values of w and its associated mse loss
    w_k = initial_w
    loss_k = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w_k))**2
    
    # stopping criterion definition:
    n_iter = 0;
    while (n_iter < max_iters):
        # computation of the searching direction by sampling one training data from the data set
        for y_b, tx_b in batch_iter(y, tx, 1):
            g = - np.transpose(tx_b).dot(y_b - tx_b.dot(w_k))
        
        # update w_kp1
        w_kp1 = w_k - gamma * g
        
        # upsate loss wrt mse cost function
        loss_kp1 = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w_kp1))**2
        
        loss_k = loss_kp1
        w_k = w_kp1
        
        # update n_iter: number of iterations
        n_iter += 1
        
    # Printing the results
    try:
        initial_w.shape[1]
        print("least_squares_SGD({bi}/{ti}): loss={l}, w = {w}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k[:,0]))
    except (IndexError, AttributeError):
        print("least_squares_SGD({bi}/{ti}): loss={l}, w = {w}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k))
        
    return w_k, loss_k









# tested
def least_squares(y, tx):
    N = tx.shape[0]
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2
    
    return w, loss







# tested
def ridge_regression(y, tx, lambda_):
    N = tx.shape[0]
    D = tx.shape[1]
    lambda_prime = lambda_ * 2 * N
    w = np.linalg.solve(np.transpose(tx).dot(tx) + lambda_prime * np.identity(D), np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    return w, loss








# tested
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    # check that initial_w has the wanted dimensions
    try:
        initial_w.shape[1]
    except IndexError:
        initial_w = np.expand_dims(initial_w, 1)
    
    # number of training data
    N = tx.shape[0]
    
    # Define initial values of w and its associated mse loss
    w_k = initial_w
    loss_k = np.sum(np.log(1 + np.exp(tx.dot(w_k))) - y * tx.dot(w_k))
    
    # stopping criterion definition:
    n_iter = 0;
    while (n_iter < max_iters):
        
        # computation of the gradient
        grad = np.transpose(tx).dot(sigmoid(tx.dot(w_k)) - y)
        
        # update w
        w_kp1 = w_k - gamma * grad
        
        # upsate loss wrt mse cost function
        loss_kp1 = np.sum(np.log(1 + np.exp(tx.dot(w_k))) - y * tx.dot(w_k))
        
        loss_k = loss_kp1
        w_k = w_kp1
        
        # update n_iter: number of iterations
        n_iter += 1
        
    # Printing the results
    try:
        initial_w.shape[1]
        print("logistic_GD({bi}/{ti}): loss={l}, w = {w}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k[:,0]))
    except (IndexError, AttributeError):
        print("logistic_GD({bi}/{ti}): loss={l}, w = {w}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k))
    return w_k, loss_k











# tested
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    # convert w to a numpy array if it is passed as a list
    if (type(initial_w) != np.ndarray):
        initial_w = np.array(initial_w)
    
    # check that initial_w has the wanted dimensions
    try:
        initial_w.shape[1]
    except IndexError:
        initial_w = np.expand_dims(initial_w, 1)
        
    
    # number of training data
    N = tx.shape[0]
    
    # Define initial values of w and its associated mse loss
    w_k = initial_w
    loss_k = np.sum(np.log(1 + np.exp(tx.dot(w_k))) - tx.dot(w_k) * y) + (lambda_ / 2) * np.linalg.norm(w_k)**2
    
    # stopping criterion definition:
    n_iter = 0;
    while (n_iter < max_iters):        
        # computation of the gradient
        grad = np.transpose(tx).dot(sigmoid(tx.dot(w_k)) - y) + lambda_ * w_k
        
        # update w
        w_kp1 = w_k - gamma * grad
        
        # upsate loss wrt mse cost function
        loss_kp1 = np.sum(np.log(1 + np.exp(tx.dot(w_k))) - tx.dot(w_k) * y) + (lambda_ / 2) * np.linalg.norm(w_k)**2
        
        loss_k = loss_kp1
        w_k = w_kp1
        
        # update n_iter: number of iterations
        n_iter += 1
        
    # Printing the results
    try:
        initial_w.shape[1]
        print("reg_logistic_GD({bi}/{ti}): loss={l}, w = {w}".format(
              bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k[:,0]))
    except (IndexError, AttributeError):
        print("reg_logistic_GD({bi}/{ti}): loss={l}, w = {w}".format(
                  bi=n_iter-1, ti=max_iters - 1, l=loss_k, w = w_k))
    return w_k, loss_k












# UTILITIES #########################################################################################################################


def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))









def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
































