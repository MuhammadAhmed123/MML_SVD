import numpy as np
# import scipy as sp

from scipy import linalg

import svd
import rsvd_n
import rsvd_s

import time

import matplotlib.pyplot as plt

matrices = []

base_size_matrix = 5000
max_size_matrix = 10001
step_size = 250

for i in range(base_size_matrix,max_size_matrix,step_size):
    random_matrix = np.random.rand(i, i)
    matrices.append(random_matrix)


np_time = []
sp_time = []
svd_time = []
rsvd_n_time = []
rsvd_s_time = []


for m in matrices:
    # numpy
    t1 = time.time()
    svd_np = np.linalg.svd(m)
    t2 = time.time()

    np_time.append(t2-t1)

    # scipy
    t3 = time.time()
    svd_sp = linalg.svd(m)
    t4 = time.time()

    sp_time.append(t4-t3)

    # # svd
    # t5 = time.time()
    # svd_svd = svd.svd(m)
    # t6 = time.time()

    # svd_time.append(t6-t5)

    # rsvd_n
    t7 = time.time()
    svd_rsvd_n = rsvd_n.rsvd(m, m.shape[0])
    t8 = time.time()

    rsvd_n_time.append(t8-t7)

    # rsvd_s
    t9 = time.time()
    svd_rsvd_s = rsvd_s.rsvd(m, m.shape[0])
    t10 = time.time()

    rsvd_s_time.append(t10-t9)

x = [i for i in range(base_size_matrix,max_size_matrix,step_size)]

# plt.plot(x, np_time, 'r', x, sp_time, 'b', x, svd_time, 'g', x, rsvd_n_time, 'y', rsvd_s_time, 'p')
# plt.show()


plt.plot(x, np_time, label='numpy')
plt.plot(x, sp_time, label='scipy')
# plt.plot(x, svd_time, label='pure_svd')
plt.plot(x, rsvd_n_time, label='rsvd_numpy')
plt.plot(x, rsvd_s_time, label='rsvd_scipy')
plt.xlabel('Matrix Dimensions label')
plt.ylabel('Time')
plt.title("SVD Analysis")
plt.legend()
plt.show()