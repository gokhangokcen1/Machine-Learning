import numpy as np
import matplotlib.pyplot as plt 

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0]
# or m = len(x_train)

# plt.scatter(x_train, y_train, marker='x', c='r')
# plt.title("Housing prices")
# plt.ylabel('Price (in 1000s of dollars)')
# plt.xlabel('Size (1000 sqft)')
# plt.show()

#w = 100
#b = 100
w, b = 200, 100

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m) #[0. 0.]
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb
    
tmp_f_wb = compute_model_output(x_train, w, b,)

plt.plot(x_train, tmp_f_wb, c='b', label='Our prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual values')
plt.title("Housing prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend() #elements: our predictions and actual values
plt.show()

# prediction

x_i = 1.2
cost_1200sqft = w * x_i + b
print(f"${cost_1200sqft:.0f} thousand dollars")


