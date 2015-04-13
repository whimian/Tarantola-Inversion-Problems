'''
Problem 2: Measuring the acceleration of Gravity
'''
import numpy as np
import matplotlib.pyplot as plt

# data
t = np.array([0.2, 0.4, 0.6, 0.8])  # unit: second
z = np.array([0.62, 0.88, 0.70, 0.15])  # unit: meter

s = 0.01  # error bound of recorded time
sigma = 0.02

g = np.r_[9: 11: 0.01]
v_0 = np.r_[3.8: 4.5: 0.01]

sigma_M = np.zeros((g.shape[0], v_0.shape[0]))

t_1 = np.r_[(t[0]-s): (t[0]+s): 0.001]
t_2 = np.r_[(t[1]-s): (t[1]+s): 0.001]
t_3 = np.r_[(t[2]-s): (t[2]+s): 0.001]
t_4 = np.r_[(t[3]-s): (t[3]+s): 0.001]

for i in range(g.shape[0]):
    for j in range(v_0.shape[0]):
        S = 0
        for k in range(len(t_1)):
            S = S + np.exp(-(
                    np.abs(v_0[j]*t_1[k] - 0.5*g[i]*t_1[k]**2 - z[0])/sigma +
                    np.abs(v_0[j]*t_2[k] - 0.5*g[i]*t_2[k]**2 - z[1])/sigma +
                    np.abs(v_0[j]*t_3[k] - 0.5*g[i]*t_3[k]**2 - z[2])/sigma +
                    np.abs(v_0[j]*t_4[k] - 0.5*g[i]*t_4[k]**2 - z[3])/sigma))

        sigma_M[i][j] = S
# draw a contour map
plt.contourf(v_0, g, sigma_M)
