
import numpy as np

def relative_l2_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.sum((x-y)**2, axis = (1,2,3))
    temp = np.sum((x)**2, axis = (1,2,3))
    return np.sqrt(mse/temp), mse, temp, np.sqrt(np.sum(mse)/np.sum(temp))

def relative_rmse_error_ornl(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    maxv = np.max(x)
    minv = np.min(x)
    return np.sqrt(mse)/(maxv - minv)

def mean_relative_rmse_error_ornl(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2, axis = 1)
    maxv = np.max(x, axis = 1)
    minv = np.min(x, axis = 1)
    return np.mean(np.sqrt(mse)/(maxv - minv))

def relative_l2_error_mgard(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.sum((x-y)**2)
    temp = np.sum((x)**2)
    return np.sqrt(mse/temp), mse, temp

def max_relative_l2_error(original_data,  recons_data, shape = [-1, 20*16*16]):
    original_data = original_data.reshape(shape)
    recons_data = recons_data.reshape(shape)
    diff = np.abs(original_data-recons_data)
    error_norm = np.linalg.norm(diff, axis=1)
    return np.max(error_norm), np.max(diff)

