
import numpy as np
import math
#import scipy.stats as ss
#import scipy.special as sspec




# EXERCISE 2 ###############################################################################################################


def compute_loss(y, tx, w, cost = "mse", lambda_ = 0):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = tx.shape[0]
    if (cost == "mse"):
        return (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2
    elif (cost == "mae"):
        return (1 / N) * np.sum(np.abs(y - tx.dot(w)))
    elif (cost == "ridge"):
        return (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    elif (cost == "logistic"):
        tx_dot_w = tx.dot(w)
        log_1p_exp = np.logaddexp(0, tx_dot_w)
        #log_1p_exp = tx_dot_w
        #log_1p_exp[log_1p_exp < 100] = np.log(1 + np.exp(log_1p_exp[log_1p_exp < 100]))        
        #for i in range(N):
        #    if (tx_dot_w[i] > 30):
        #        log_1p_exp[i] = tx_dot_w[i]
        #    else:
        #        log_1p_exp[i] = np.log(1 + np.exp(tx_dot_w[i]))
        return np.sum(log_1p_exp - y * tx_dot_w)
    
    elif (cost == "reg_logistic"):
        tx_dot_w = tx.dot(w)
        log_1p_exp = np.logaddexp(0, tx_dot_w)
        #log_1p_exp[log_1p_exp < 100] = np.log(1 + np.exp(log_1p_exp[log_1p_exp < 100]))
        #log_1p_exp = np.zeros((N,1))
        #for i in range(N):
        #    if (tx_dot_w[i] > 30):
        #        log_1p_exp[i] = tx_dot_w[i]
        #    else:
        #        log_1p_exp[i] = np.log(1 + np.exp(tx_dot_w[i]))
        return np.sum(log_1p_exp - tx_dot_w * y) + (lambda_ / 2) * np.linalg.norm(w)**2
    else:
        raise IllegalArgument("Invalid cost argument in compute_loss")
    return 0






def grid_search(y, tx, w0, w1, cost = "mse", lambda_ = 0):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i, j] = compute_loss(y, tx, np.array([w0[i], w1[j]]))
    
    argmin_index = np.argmin(losses);
    best_i = argmin_index // len(w1)
    best_j = argmin_index % len(w1)
    
    # best_i, best_j = np.unravel_index(np.argmin(losses), losses.shape)
    return [w0[best_i], w1[best_j]], losses




def compute_gradient(y, tx, w, cost = "mse", lambda_ = 0):
    """Compute the gradient."""
    N = tx.shape[0];
    if (cost == "mse"):
        return -(1/N) * np.transpose(tx).dot(y - tx.dot(w))
    elif (cost == "mae"):
        # Note that here it is not a gradient strictly speaking because the mae is not differentiable everywhere
        error_vec = y - tx.dot(w)
        sub_grad = [0 if error_vec[i] == 0 else error_vec[i] / np.abs(error_vec[i]) for i in range(N)]
        return -(1 / N) * np.transpose(tx).dot(sub_grad)
    elif (cost == "ridge"):
        return -(1/N) * np.transpose(tx).dot(y - tx.dot(w)) + 2 * lambda_ * w
    elif (cost == "logistic"):
        try:
            w.shape[1]
        except IndexError:
            w = np.expand_dims(w, 1)
        return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)
    elif (cost == "reg_logistic"):
        try:
            w.shape[1]
        except IndexError:
            w = np.expand_dims(w, 1)
        return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y) + lambda_ * w
    else:
        raise IllegalArgument("Invalid cost argument in compute_gradient function.")
    return 0




# Be careful, gamma of 2 for MSE doesn't converge
def gradient_descent(y, tx, initial_w, max_iters, gamma, cost ="mse", lambda_ = 0, tol = 1e-6, thresh_test_div = 100, update_gamma = False):
    """Gradient descent algorithm."""
    if (cost not in ["mse", "mae", "logistic", "reg_logistic", "ridge"]):
        raise IllegalArgument("Invalid cost argument in gradient_descent function")
    
    # convert w to a numpy array if it is passed as a list
    if (type(initial_w) != np.ndarray):
        initial_w = np.array(initial_w)
    
    # ensure that w_initial is formatted in the right way if implementing logistic regression
    if (cost == "logistic" or cost == "reg_logistic"):
        try:
            initial_w.shape[1]
        except IndexError:
            initial_w = np.expand_dims(initial_w, 1)
    
        
    # Define parameters to store w and loss
    loss_k = compute_loss(y, tx, initial_w, cost = cost, lambda_ = lambda_)
    w_k = initial_w
    
    # test_div is an integer that is incremented for each consecutive increase in loss.
    # If test_div reaches thresh_test_div the programm stops and divergence is assumed
    test_div = 0
    dist_succ_loss = loss_k * tol + 1
    n_iter = 0;
    gamma_init = gamma
    
    c = 0.5
    
    while (n_iter < max_iters and dist_succ_loss > tol * loss_k):
        gamma = gamma_init
        
        grad = compute_gradient(y, tx, w_k, cost = cost, lambda_ = lambda_)
        
        loss_wk_pdir = compute_loss(y, tx, w_k - gamma * grad, cost = cost, lambda_ = lambda_)
        norm_grad_2 = np.linalg.norm(grad)**2
        while (loss_wk_pdir > loss_k - c * gamma * norm_grad_2):
            gamma = gamma / 100
            loss_wk_pdir = compute_loss(y, tx, w_k - gamma * grad, cost = cost, lambda_ = lambda_)
        # updating w
        w_kp1 = w_k - gamma * grad
            
        #print(np.linalg.norm(gamma * grad))
        loss_kp1 = compute_loss(y, tx, w_kp1, cost = cost, lambda_ = lambda_)
        
        # Test of divergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss_kp1 > loss_k):
            test_div += 1
            if (test_div >= thresh_test_div):
                print("Stopped computing at iteration {n_iter}".format(n_iter = n_iter))
                print("because {thresh_test_div} consecutive iterations have involved an increase in loss.".format(thresh_test_div = thresh_test_div))
                return w_kp1, loss_kp1
        else:
            test_div = 0
     
        # update distance between two successive w
        dist_succ_loss = np.abs(loss_kp1 - loss_k)
        
        # update n_iter: number of iterations
        n_iter += 1
        
        w_k = w_kp1
        loss_k = loss_kp1
        
        # update gamma if update_gamma is True
        if (update_gamma == True):
            gamma = (1 / (1 + n_iter)) * gamma
    
    # Printing the results
    try:
        initial_w.shape[1]
        print("GD({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
              bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k[:,0]))
    except (IndexError, AttributeError):
        print("GD({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
                  bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k))
    return w_k, loss_k






def compute_stoch_gradient(y, tx, w, batch_size=1, cost ="mse"):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    batched = [x for x in batch_iter(y, tx, batch_size)][0]
    y_b = batched[0]
    tx_b = batched[1]

    if (cost == "mse"):
        return(-1 / batch_size) * np.transpose(tx_b).dot(y_b - tx_b.dot(w))
    elif (cost == "mae"):
        error_vec = y_b - tx_b.dot(w)
        sub_grad_abs = [0 if error_vec[i] == 0 else error_vec[i] / np.abs(error_vec[i]) for i in range(batch_size)]
        return -(1 / batch_size) * np.transpose(tx_b).dot(sub_grad_abs)
    else:
        raise IllegalArgument("Invalid cost argument in compute_stoch_gradient")
    
    return 0


# batxh_iter is used to sample the training data used to compute the stochastic gradient
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




def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, cost = "mse", tol = 1e-6, thresh_test_div = 100, update_gamma = False):
    
    if (cost not in ["mse", "mae"]):
        raise IllegalArgument("Invalid cost argument in stochastic_gradient_descent function")
    
    # convert w to a numpy array if it is passed as a list
    if (type(initial_w) != np.ndarray):
        initial_w = np.array(initial_w)
    
    # Define parameters to store w and loss
    w_k = initial_w
    loss_k = compute_loss(y, tx, w_k, cost)
    
    # test_conv is an integer that is incremented for each consecutive increase in loss.
    # If test_conv reaches thresh_test_div the programm stops and divergence is assumed
    test_div = 0
    
    dist_succ_w = tol + 1
    n_iter = 0
    while (n_iter < max_iters and dist_succ_w > tol):
        stoch_grad = compute_stoch_gradient(y, tx, w_k, batch_size, cost)
        
        # updating w
        w_kp1 = w_k - gamma * stoch_grad
        loss_kp1 = compute_loss(y, tx, w_kp1, cost)
        
        # Test of convergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss_kp1 > loss_k):
            test_div += 1
            if (test_div == thresh_test_div):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return w_kp1, loss_kp1
        else:
            test_div = 0
        
        # computing distance between new w and former one
        dist_succ_w = np.linalg.norm(loss_kp1 - loss_k)
        
        w_k = w_kp1
        loss_k = loss_kp1
        
        # updating the number of iteration
        n_iter += 1
        
        # update gamma if update_gamma is True
        if (update_gamma == True):
            gamma = (1 / (1 + n_iter)) * gamma

    try:
        initial_w.shape[1]
        print("SGD({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
              bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k[:,0]))
    except (IndexError, AttributeError):
        print("SGD({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
                  bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k))
    
    return w_k, loss_k



# EXERCISE 3 ###################################################################################################################



def least_squares(y, tx):
    """calculate the least squares solution."""
    N = tx.shape[0]
    w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2
    
    return w, loss



# Build the feature matrix with polynomial basis
def build_poly(x, degree):
    """
    Construct a polynomial basis function whose degree for each dimension is defined by degree.
    
    return tx
    """
    # Exception handling if x has only one column: cannot call x.shape[1]
    try:
        D = x.shape[1]
    except IndexError:
        D = 1
        
    if (x.shape[1] != len(degree)):
        raise IncompatibleDimensons("The dimensions of the data matrix x and of the degree vector do not match")
        
    N = x.shape[0]
    D = x.shape[1]
    
    # First column is offset column of 1
    tx = np.ones((N,1))
    for dim in range(D):
        if (type(degree[dim]) != int and type(degree[dim]) != np.int64):
            for deg in range(int(degree[dim])):
                tx = np.c_[tx, x[:,dim]**(deg+1)]
        else:
            for deg in range(degree[dim]):
                tx = np.c_[tx, x[:,dim]**(deg+1)]
        
            
    return tx






def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    returns [x_train, y_train], [x_val, y_val]
    """
    # set seed
    np.random.seed(seed)
    
    N = x.shape[0]
    
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    split_index = round(N * ratio);
    
    y_train = shuffled_y[:split_index]
    x_train = shuffled_x[:split_index]
    
    y_val = shuffled_y[split_index:]
    x_val = shuffled_x[split_index:]
    
    return [x_train, y_train], [x_val, y_val]






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




def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    lambda_prime = lambda_ * 2 * N
    I = np.diag(np.ones(D))
    w = np.linalg.solve(np.transpose(tx).dot(tx) + lambda_prime * I, np.transpose(tx).dot(y))
    loss = (1 / (2*N)) * np.linalg.norm(y - tx.dot(w))**2 + lambda_ * np.linalg.norm(w)**2
    return w, loss






# EXERCISE 4 #############################################################################################################

def build_k_indices(y, k_fold, seed):
    """
    build k indices for k-fold.
    return np.array(k_indices)
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, degree, lambda_ = 0, max_iters = 1000, gamma = 1, cost = "mse", tol = 1e-6, thresh_test_div = 10, update_gamma = False, newton_flag = False):
    """
    k_indices is the a list whose row k is the set of indices corresponding to the kth partition of the data set
    k is the index indicating which partition of the dataset to use as a the testing set
    degree is a vector indicating which polynomial degree to use for the modelling of each dimension. The length of dergree
        must correspond to the number of dimension of x
    
    return the loss of ridge regression.
    k_indices defines the partition of the data.
    k defines the index of the data partition that is used for testing.
    algo: defines which algorithm to use to fit the data.
        "ls" use least squares.
        "ridge" use ridge regression
    return 
    """
    
    test_indices = k_indices[k]
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    train_indices = k_indices[[x for x in range(k_indices.shape[0]) if x != k]]
    x_train = x[np.ravel(train_indices)]
    y_train = y[np.ravel(train_indices)]

    
    tx_test = build_poly(x_test, degree)
    tx_train = build_poly(x_train, degree)
    
    if (cost == "mse"):
        w_opti, mse_train_loss = least_squares(y_train, tx_train)
        train_loss = compute_loss(y_train, tx_train, w_opti, cost = "mse")
        test_loss = compute_loss(y_test, tx_test, w_opti, cost = "mse")
        return train_loss, test_loss
    
    elif (cost == "ridge"):
        w_opti, ridge_train_loss = ridge_regression(y_train, tx_train, lambda_)
        train_loss = compute_loss(y_train, tx_train, w_opti, cost = "mse")
        test_loss = compute_loss(y_test, tx_test, w_opti, cost = "mse")
        return train_loss, test_loss
    
    
    elif (cost == "logistic"):
        initial_w = np.zeros((tx_train.shape[1], 1))
        if (newton_flag == True):
            w_opti, logistic_loss = newton(y_train, tx_train, initial_w, max_iters, gamma, cost='logistic', tol=tol, thresh_test_div=thresh_test_div, update_gamma=update_gamma)
        else:
            w_opti, logistic_loss = gradient_descent(y_train, tx_train, initial_w, max_iters, gamma, cost='logistic', tol=tol, thresh_test_div=thresh_test_div, update_gamma=update_gamma)
            
        train_loss = compute_loss(y_train, tx_train, w_opti, cost = 'logistic')
        test_loss = compute_loss(y_test, tx_test, w_opti, cost = 'logistic')
        return train_loss, test_loss
    
    
    elif (cost == "reg_logistic"):
        initial_w = np.zeros((tx_train.shape[1], 1))
        
        if (newton_flag == True):
            w_opti, logistic_loss = newton(y_train, tx_train, initial_w, max_iters, gamma, cost='reg_logistic', tol=tol, thresh_test_div=thresh_test_div, update_gamma=update_gamma)
        else:
            w_opti, reg_logistic_loss = gradient_descent(y_train, tx_train, initial_w, max_iters, gamma, cost='reg_logistic', lambda_=lambda_, tol=tol, thresh_test_div=thresh_test_div, update_gamma=update_gamma)
            
        train_loss = compute_loss(y_train, tx_train, w_opti, cost = "logistic")
        test_loss = compute_loss(y_test, tx_test, w_opti, cost = "logistic")
        return train_loss, test_loss
    
    else:
        raise IllegalArgument("Invalid cost argument in cross validation function")
    return train_loss, test_loss


def tuner_degree_lambda(y, x, degree_min, degree_max, lambda_min = -4, lambda_max = 0, nb_lambda = 30, k_fold=4, seeds=[1], max_iters=1000, gamma=1, cost="mse", tol=1e-6, thresh_test_div=10, update_gamma=False):
    """
    degree_min is a vector that contains the minimum degree to investigate for each dimension of x
    degree_max is a vector that contains the maximum degree to investigate for each dimension of x
    k_fold is the number of divisions used for cross-validation
    
    return best_degree, best_lambda, min_rmse_te, min_rmse_tr
    
    Finds the degree and lambda that yield the less test rmse thanks to cross validation 
    for a particular partition of the data defined by seed.
    Be careful that we are only working here a a single partition of the data and that it may not be representative for the choice
    of lambda and of the degree. 
    Might be a good idea to compute the rmse over several seeds and to look for the best lambda and degree.
    Take a look at averaged_cross_validation_best_degree_lambda for a function 
    looking at an averaged test rmse.
    degrees must be n array containing the degrees that are tested.
    lambda_min, lambda_max and nb_lambda define the interval of test for lambda.
    plot is a boolean allowing the function to plot the rmse
    return best_degree, best_lambda
    """
    lambdas = np.logspace(lambda_min, lambda_max, nb_lambda)
    
    
    dim = x.shape[1]
    
    if (type(degree_min) == np.ndarray):
        degree_min = degree_min.tolist()
    if (type(degree_max) == np.ndarray):
        degree_max = degree_max.tolist()
    
    deg_min_max = np.c_[degree_min, degree_max]
    deg_range = [deg_min_max[x][1] - deg_min_max[x][0] + 1 for x in range(dim)]
    tot_nb_combinations = 1
    for i in range(dim):
        tot_nb_combinations *= deg_range[i]

    degrees = np.zeros((tot_nb_combinations, dim))

    left_index = 0
    right_index = degrees.shape[0]-1
    curr_dim = 0
    fill_deg(degrees, left_index, right_index, curr_dim, deg_min_max)
    
    
    rmse_te = np.empty((len(seeds), len(degrees), len(lambdas)))
    rmse_tr = np.empty((len(seeds), len(degrees), len(lambdas)))
    
    min_rmse_te = math.inf
    for ind_seed, seed_elem in enumerate(seeds):
        print("# SEED :", ind_seed, "##############################################################")
        # split data in k fold
        k_indices = build_k_indices(y, k_fold, seed_elem)
        
        # degrees is the list of all the degree array that are tested
        for ind_deg, deg in enumerate(degrees):
            # define lists to store the loss of training data and test data
            for ind_lambda_, lambda_ in enumerate(lambdas):
                tmp_rmse_tr = []
                tmp_rmse_te = []
                for k in range(k_fold):
                    print(deg)
                    print("Lambda:", lambda_)
                    print("{k}/{k_fold}".format(k=k, k_fold=k_fold))
                    # The initial_w used to execute the gradient descent is defined in the cross_validation function
                    train_loss, test_loss = cross_validation(y, x, k_indices, k, deg, lambda_ = lambda_, max_iters = max_iters,\
                                                             gamma = gamma, cost = cost, tol = tol, thresh_test_div = thresh_test_div,\
                                                             update_gamma = update_gamma)
                    tmp_rmse_tr += [np.sqrt(2 * train_loss)]
                    tmp_rmse_te += [np.sqrt(2 * test_loss)]

                rmse_tr[ind_seed, ind_deg, ind_lambda_] = sum(tmp_rmse_tr) / k_fold
                rmse_te[ind_seed, ind_deg, ind_lambda_] = sum(tmp_rmse_te) / k_fold
                print("rmse_tr:", rmse_tr[ind_seed, ind_deg, ind_lambda_])
                print("rmse_te:", rmse_te[ind_seed, ind_deg, ind_lambda_])

                    
    rmse_te_mean = np.mean(rmse_te, axis=0)
    rmse_tr_mean = np.mean(rmse_tr, axis=0)
    
    rmse_te_std = np.std(rmse_te, axis = 0)
    rmse_tr_std = np.std(rmse_tr, axis = 0)
    
    best_index = np.argmin(rmse_te_mean)
    best_ind_d = best_index // len(lambdas)
    best_ind_lambda = best_index % len(lambdas)
    best_degree = degrees[best_ind_d]
    best_lambda = lambdas[best_ind_lambda]  
                    
    return (best_degree, best_ind_d), (best_lambda, best_ind_lambda), (rmse_te_mean, rmse_te_std), (rmse_tr_mean, rmse_tr_std)


        
        
def fill_deg(A, left, right, curr_dim, deg_min_max):
    """
    Function that outputs a matrix whose rows are every degree combination to test
    """
    if (curr_dim < A.shape[1]):
        
        min_deg = deg_min_max[curr_dim][0]
        max_deg = deg_min_max[curr_dim][1]
        if (type(min_deg) == np.float64 and type(max_deg) == np.float64):
            deg_range = range(min_deg.astype(int), max_deg.astype(int)+1)
        else:
            deg_range = range(min_deg, max_deg+1)
        nb_deg_test = max_deg - min_deg + 1
        batch_size = int((right - left + 1) / nb_deg_test)

        for i in range(nb_deg_test):
            ind_1 = left + i * batch_size
            ind_2 = left + (i+1) * batch_size
            A[ind_1:ind_2, curr_dim] = deg_range[i]
            fill_deg(A, ind_1, ind_2 - 1, curr_dim + 1, deg_min_max)  
        
        
        
        
        


def averaged_cross_validation_best_degree_lambda(y, x, degrees, k_fold, nb_seed = 100, \
                                                 lambda_min = -4, lambda_max = 0, nb_lambda = 30, plot = False):
    lambdas = np.logspace(lambda_min, lambda_max, nb_lambda)
    
    seeds = range(nb_seed)
    
    # define list to store the variable
    rmse_te = np.empty((len(seeds), len(degrees), len(lambdas)))    
    
    for ind_seed, seed in enumerate(seeds):
        k_indices = build_k_indices(y, k_fold, seed)
        for ind_d, d in enumerate(degrees):
            for ind_lambda_, lambda_ in enumerate(lambdas):
                tmp_rmse_te = []
                for k in range(k_fold):
                    loss_test = cross_validation(y, x, k_indices, k, lambda_, d)[1]
                    tmp_rmse_te += [np.sqrt(2 * loss_test)]

                rmse_te[ind_seed, ind_d, ind_lambda_] = sum(tmp_rmse_te) / k_fold


    rmse_te_mean = np.mean(rmse_te, axis=0)
    
    best_index = np.argmin(rmse_te_mean)
    best_ind_d = best_index // len(lambdas)
    best_ind_lambda = best_index % len(lambdas)
    
    if (plot == True):
        plt.plot(degrees, rmse_te_mean[:, best_ind_lambda], marker=".", color='r', label='test error')
        plt.xlabel("degree")
        plt.ylabel("rmse")
        plt.title("cross validation")
        plt.legend(loc=2)
        plt.grid(True)
        plt.savefig("cross_validation")

        plt.figure()
        plt.semilogx(lambdas, rmse_te_mean[best_ind_d], marker=".", color='r', label='test error')
        plt.xlabel("lambda")
        plt.ylabel("rmse")
        plt.title("cross validation")
        plt.legend(loc=2)
        plt.grid(True)
        plt.savefig("cross_validation")
    
    best_degree = degrees[best_ind_d]
    best_lambda = lambdas[best_ind_lambda]
    
    return best_degree, best_lambda



# EXERCISE 5 #################################################################################################################

def sigmoid(t):
    """apply sigmoid function on t."""
    #return sspec.expit(t)
    return 1 / (1 + np.exp(-t))



def compute_hessian(y, tx, w, cost = "mse", lambda_ = 0):
    N = tx.shape[0]
    if (cost == "mse"):
        return (1 / N) * np.transpose(tx).dot(tx)
    elif (cost == "logistic"):
        S_diag = sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w)))
        S = np.diag(np.squeeze(S_diag))
        return np.transpose(tx).dot(S).dot(tx)
    elif (cost == "reg_logistic"):
        D = tx.shape[1]
        S_diag = sigmoid(tx.dot(w))
        S = np.diag(np.squeeze(S_diag))
        return np.transpose(tx).dot(S).dot(tx) + np.identity(D) * lambda_
    else:
        raise IllegalArgument("Invalid cost argument in compute_hessian")
    return 0




def newton(y, tx, initial_w, max_iters, gamma, cost ="mse", lambda_ = 0, tol = 1e-6, thresh_test_div = 10, update_gamma = False):
    """
    Newton method.
    Think to change the output to only return single values and not entire arrays
    """
    if (cost not in ["mse", "logistic", "reg_logistic"]):
        raise IllegalArgument("Invalid cost argument in gradient_descent function")
    
    # ensure that w_initial is formatted in the right way if implementing logistic regression
    if (cost == "logistic" or cost == "reg_logistic"):
        try:
            initial_w.shape[1]
        except IndexError:
            initial_w = np.expand_dims(initial_w, 1)
    
        
    # Define parameters to store w and loss
    w_k = initial_w
    loss_k = compute_loss(y, tx, w_k, cost = cost, lambda_ = lambda_)
    
    test_div = 0
    dist_succ_loss = tol + 1
    n_iter = 0;
    while (n_iter < max_iters and dist_succ_loss > tol):
        grad = compute_gradient(y, tx, w_k, cost = cost, lambda_ = lambda_)
        hess = compute_hessian(y, tx, w_k, cost = cost)
        
        # updating w
        z = np.linalg.solve(hess, -gamma * grad)
        w_kp1 = z + w_k
        loss_kp1 = compute_loss(y, tx, w_kp1, cost = cost, lambda_ = lambda_)
        
        # Test of divergence, test_conv counts the number of consecutive iterations for which loss has increased
        if (loss_kp1 > loss_k):
            test_div += 1
            if (test_div >= thresh_test_div):
                print("Stopped computing because 10 consecutive iterations have involved an increase in loss.")
                return w_kp1, loss_kp1
        else:
            test_div = 0

     
        # update distance between two successive w
        dist_succ_loss = np.abs(loss_k - loss_kp1)
        
        w_k = w_kp1
        loss_k = loss_kp1
        
        # update n_iter: number of iterations
        n_iter += 1
        
        # update gamma if update_gamma is True
        if (update_gamma == True):
            gamma = 1 / (1 + n_iter) * gamma
    
    # Printing the results
    try:
        initial_w.shape[1]
        print("Newton({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
              bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k[:,0]))
    except (IndexError, AttributeError):
        print("Newton({bi}/{ti}), cost: {cost}: loss={l}, w={w}".format(
                  bi=n_iter-1, ti=max_iters - 1, cost = cost, l=loss_k, w=w_k))
    return w_k, loss_k





    
    
# EXCEPTIONS ##################################################################################################################
    
class IllegalArgument(Exception):
    pass

class UndefinedArgument(Exception):
    pass

