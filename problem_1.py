'''
Estination of the epicentral coordinates of a seismic event
'''
import numpy as np
import matplotlib.pyplot as plt

# data
std_d = 0.10  # standard deviation
v = 5
station = np.array([(3, 15), (3, 16), (4, 15), (4, 16), (5, 15), (5, 16)])
t_obs = np.array([3.12, 3.26, 2.98, 3.12, 2.84, 2.98])

X = np.r_[0:20:201j]
Y = np.linspace(0, 20, 201)

# calculate the posterior distribution
sigma_M = np.zeros((len(X), len(Y)))

for i in range(X.shape[0]):
    for j in range(Y.shape[0]):
        x = X[i]
        y = Y[j]
        t_cal = np.zeros(t_obs.shape)
        for k in range(t_cal.shape[0]):
            t_cal[k] = ((x - station[k][0])**2 +
                        (y - station[k][1])**2)**0.5 / v
        sigma = 0
        for k in range(t_obs.shape[0]):
            sigma = sigma + (t_cal[k] - t_obs[k])**2
        sigma_M[j, i] = np.exp(sigma/(-2 * std_d**2))

# making a graph

# plt.imshow(sigma_M, origin=[0,0])
plt.contourf(Y, X, sigma_M,)
xx = [station[0][0], station[1][0], station[2][0],
      station[3][0], station[4][0], station[5][0]]
yy = [station[0][1], station[1][1], station[2][1],
      station[3][1], station[4][1], station[5][1]]
plt.plot(xx, yy, 'ro')
plt.show()
