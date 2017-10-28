import numpy as np
import matplotlib.pyplot as plt

#Returns all the rows and the y values of the dataset that does not contains -999 values
def get_clean_data(tx,y):
    tx_clean = []
    y_clean = []
    y2 = y.reshape([y.shape[0],1])
    for i in range(tx.shape[0]):
        has_a_nan = 0
        for elem in tx[i]:
            if (elem == -999):
                has_a_nan = 1
        if (has_a_nan ==0):
            tx_clean.append(tx[i].reshape([1,30]))
            y_clean.append(y2[i])

    tx_clean = np.concatenate(tx_clean,axis=0)
    y_clean = np.concatenate(y_clean).reshape([tx_clean.shape[0],1])
    return (tx_clean,y_clean)

#Returns the standardized dataset, lets the -999 values
def standardize(tx):
    mean = np.zeros([tx.shape[1],1])
    std = np.zeros([tx.shape[1],1])
    txstd = np.zeros([tx.shape[0],tx.shape[1]])
    for i in range(tx.shape[1]):
        txi = tx[:,i]
        txi_filtered = txi[txi != -999.]
        txi_filtered_mean = txi_filtered.mean()
        txi_filtered_std = txi_filtered.std()
        for j in range(tx.shape[0]):
            if (tx[j,i] != -999):
                txstd[j,i] = (tx[j,i] - txi_filtered_mean)/txi_filtered_std
    return txstd

#Returns two sets of data, corresponding to each label (-1 or 1)
#first element returned is +1, second is -1 labeled data
def differentiate_two_categories(tx, y):
    tx_p1 = []
    tx_m1 = []
    for i in range(tx.shape[0]):
        if (y[i] == -1):
            tx_m1.append(tx[i].reshape([1,30]))
        if (y[i] == 1):
            tx_p1.append(tx[i].reshape([1,30]))

    tx_p1 = np.concatenate(tx_p1,axis=0)
    tx_m1 = np.concatenate(tx_m1,axis=0)
    return (tx_p1, tx_m1)

#Returns repartition histogram in function of a feature.
#The histogram is normalized
def plothisto(tx_m1, tx_p1, n_feature, save_path = 'None'):
    plt.figure(figsize=[7,7])
    plt.hist(tx_m1[:,n_feature], bins = 100, alpha=0.5, label = '-1', normed=True)
    plt.hist(tx_p1[:,n_feature], bins = 100, alpha=0.5, label = '1', normed=True)
    plt.legend(loc='upper right')
    plt.title('feature ' + str(n_feature) + ' repartition')
    if (save_path != 'None'):
        plt.savefig(save_path + 'feature_' + str(n_feature) + '_repartition')
       

def pass_data_to_zero_one(y):
    y_zero_one = np.ones(y.shape[0])
    for i in range(y.shape[0]):
        if(y[i] == -1):
            y_zero_one[i] = 0
    return y_zero_one

def pass_data_to_minus_one_one(y):
    y_minus_one_one = np.ones(y.shape[0])
    for i in range(y.shape[0]):
        if(y[i] == 0):
            y_minus_one_one[i] = -1
    return y_minus_one_one

def put_nan_to_mean(tx, y):
    tx_nan_to_mean = tx
    (tx_clean, y_clean) = get_clean_data(tx, y)
    means = tx_clean.mean(axis=0)
    for i in range(tx.shape[0]):
        for j in range(tx.shape[1]):
            if(tx[i,j] == -999.0):
                tx_nan_to_mean[i,j] = means[j]
    return tx_nan_to_mean

