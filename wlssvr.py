"""
Weighted Least Squares Support Vector Machine (WLSSVM) Implementation

This module implements WLSSVM for regression tasks, including data loading,
kernel computation, model training, and prediction.
"""

import numpy as np

def load_dataset(filename):
    """Load data from a file."""
    with open(filename, 'r') as file:
        data = [list(map(float, line.strip().split('\t'))) for line in file]
    return np.mat(data)[:, 0].T, np.mat(data)[:, 1].T

def kernel_trans(X, A, k_tup):
    """Compute kernel transformation."""
    m = X.shape[0]
    K = np.zeros((m, 1))
    
    if k_tup[0] == 'lin':
        K = X * A.T
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j] - A
            K[j] = delta_row * delta_row.T
        K = np.exp(K / (-1 * k_tup[1] ** 2))
    else:
        raise ValueError(f"Unsupported kernel type: {k_tup[0]}")
    
    return K

class OptStruct:
    """Optimization structure for WLSSVM."""
    def __init__(self, data_mat_in, class_labels, C, k_tup):
        self.X = data_mat_in
        self.label_mat = class_labels
        self.C = C
        self.m = data_mat_in.shape[0]
        self.alphas = np.zeros((self.m, 1))
        self.b = 0
        self.K = np.zeros((self.m, self.m))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)

def least_squares(data_mat_in, class_labels, C, k_tup):
    """Perform least squares optimization."""
    os = OptStruct(data_mat_in, class_labels, C, k_tup)
    unit = np.ones((os.m, 1))
    I = np.eye(os.m)
    zero = np.zeros((1, 1))
    up_mat = np.hstack((zero, unit.T))
    down_mat = np.hstack((unit, os.K + I / float(C)))
    complete_mat = np.vstack((up_mat, down_mat))
    right_mat = np.vstack((zero, os.label_mat))
    b_alpha = np.linalg.inv(complete_mat) * right_mat
    os.b = b_alpha[0, 0]
    os.alphas = b_alpha[1:, 0]
    e = os.alphas / C
    return os.alphas, os.b, e

def weights(e):
    """Calculate weights based on error."""
    c1, c2 = 2.5, 3
    m = e.shape[0]
    v = np.zeros((m, 1))
    v1 = np.eye(m)
    q1, q3 = int(m / 4.0), int((m * 3.0) / 4.0)
    e_sorted = sorted(e.flatten().tolist()[0])
    IQR = e_sorted[q3] - e_sorted[q1]
    s = IQR / (2 * 0.6745)
    shang = np.abs(e / s)
    
    for x in range(m):
        if shang[x, 0] <= c1:
            v[x, 0] = 1.0
        elif c1 < shang[x, 0] <= c2:
            v[x, 0] = (c2 - shang[x, 0]) / (c2 - c1)
        else:
            v[x, 0] = 0.0001
        v1[x, x] = 1 / float(v[x, 0])
    
    return v1

def weighted_least_squares(data_mat_in, class_labels, C, k_tup, v1):
    """Perform weighted least squares optimization."""
    os = OptStruct(data_mat_in, class_labels, C, k_tup)
    unit = np.ones((os.m, 1))
    zero = np.zeros((1, 1))
    up_mat = np.hstack((zero, unit.T))
    down_mat = np.hstack((unit, os.K + v1 / float(C)))
    complete_mat = np.vstack((up_mat, down_mat))
    right_mat = np.vstack((zero, os.label_mat))
    b_alpha = np.linalg.inv(complete_mat) * right_mat
    os.b = b_alpha[0, 0]
    os.alphas = b_alpha[1:, 0]
    return os.alphas, os.b

def predict(alphas, b, data_mat, k_tup):
    """Make predictions using the trained model."""
    m = data_mat.shape[0]
    predict_result = np.zeros((m, 1))
    for i in range(m):
        Kx = kernel_trans(data_mat, data_mat[i, :], k_tup)
        predict_result[i, 0] = Kx.T * alphas + b
    return predict_result

def predict_average_error(predict_result, label):
    """Calculate average prediction error."""
    return np.mean(np.abs(predict_result - label))

if __name__ == '__main__':
    print('--------------------Load Data------------------------')
    data_mat, label_mat = load_dataset('sine.txt')
    
    print('--------------------Parameter Setup------------------')
    C = 0.6
    k1 = 0.3
    kernel = 'rbf'
    k_tup = (kernel, k1)
    
    print('-------------------Save LSSVM Model-----------------')
    alphas, b, e = least_squares(data_mat, label_mat, C, k_tup)
    
    print('----------------Calculate Error Weights-------------')
    v1 = weights(e)
    
    print('------------------Save WLSSVM Model-----------------')
    alphas1, b1 = weighted_least_squares(data_mat, label_mat, C, k_tup, v1)
    
    print('------------------Predict Result--------------------')
    predict_result = predict(alphas1, b1, data_mat, k_tup)
    
    print('-------------------Average Error--------------------')
    average_error = predict_average_error(predict_result, label_mat)
    print(f"Average Error: {average_error}")