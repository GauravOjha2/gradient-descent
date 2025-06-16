import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 3, 4, 5, 6])

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m   

    return dj_dw, dj_db

# Call the function
dj_dw, dj_db = compute_gradient(x_train, y_train, w=0, b=0)
print(f'dj_dw: {dj_dw}, dj_db: {dj_db}')

#w_in=intial value of w 
#b_in=intial value of b
#num_iters=no. of iteration to run gd
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    


    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w , b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(x, y, w, b)
        J_history.append(cost)
        p_history.append([w, b])

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i}: Cost {cost:.4f}, w: {w:.4f}, b: {b:.4f}")

    return w, b, J_history, p_history

w_init = 0
b_init = 0
iterations = 10000
tmp_alpha = 1.0e-2
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")









