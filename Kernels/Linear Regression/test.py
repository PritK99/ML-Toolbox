import numpy as np

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-(np.linalg.norm(x-y)**2) / (2 * (sigma**2)))

def compute_output(alpha, X, X_train, gamma):

  N = len(X_train)  # number of train items
  sum = 0.0
  for i in range(N):
    sum += alpha[i]*rbf_kernel(X, X_train[i], gamma)  # kernel value

  return sum

def error(w, X, y, gamma):
  # mean squared error over entire training X data
  sum = 0.0
  N = len(X)
  for i in range(N):
    x = X[i]  # current data item
    y_pred = compute_output(w, x, X, gamma)
    y_actual = y[i]
    sum += (y_pred - y_actual) * (y_pred - y_actual)
  return sum / N

# -----------------------------------------------------------

def train(X, y, gamma, lrn_rate, max_epochs, alpha):
  # compute and return weights for KRR.
  # gamma is for RBF, alpha for weight decay regularization
  N = len(X)  # number training items
  indices = np.arange(N)
  w = np.random.random(size=(N))  # init wts to small rnds

  for epoch in range(max_epochs):
    np.random.shuffle(indices)  # process in random order
    for i in range(N):
      idx = indices[i]
      x = X[idx]  # get an x train item 
      y_pred = compute_output(w, x, X, gamma)
      y_actual = y[idx]
      # delta = y_pred - y_actual  # output - target form

      # adjust curr wt using pseudo gradient descent
      w[idx] +=  lrn_rate * (y_actual - y_pred)  # t-o form

      # decay weight towards zero - almost L2/ridge regular.
      w[idx] -= alpha * w[idx]  # works regardless if + or -

    # show error 5 times
    interval = max_epochs // 5
    if epoch % interval == 0:
      mse = error(w, X, y, gamma)
      print("epoch = %8d  mse = %0.4f " % (epoch, mse))
  return w

# -----------------------------------------------------------

print("\nBegin poor man's kernel ridge regression demo ")
np.random.seed(1)

X = np.array([[1,3], [2,3], [3,2], [4,2], [5,1]],
  dtype=np.float64)
y = np.array([2, 4, 1, 5, 3], dtype=np.float64)

print("\nX values: ")
print(X)
print("\nTarget y values: ")
print(y)

print("\nStarting training using pseudo gradient descent ")
gma = 0.10  # gamma for RBF()
wts = train(X, y, gamma=gma, lrn_rate=0.01,
  max_epochs=20000, alpha = 0.00001)
print("Done ")

np.set_printoptions(precision=4, suppress=True)
print("\nTrained weights = ")
print(wts)

print("\nActual y values: ")
print(y)

print("\nPredicted y values: ")
for i in range(len(X)):
  x = X[i]
  y_pred = compute_output(wts, x, X, gamma=gma)
  print("%0.2f" % y_pred)

print("\nEnd demo ")