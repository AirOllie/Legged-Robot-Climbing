import os
import time

import numpy as np
import math
import copy

class KMP:
    def __init__(self, data_refx, lenx, inDimx, outDimx, khx, lamdax, pvFlagx):
        self.data = data_refx
        self.len = lenx
        self.inDim = inDimx
        self.outDim = outDimx
        self.kh = khx
        self.lamda = lamdax
        self.pvFlag = pvFlagx

        self.W = np.zeros([self.len * self.outDim,1])

    ### to deal with high-dim input, the a, b should be high-dim variables.
    ### Here, only the one-dim variable, usually, it refers to time input (t)
    def kernel_extend(self, a, b):
        kernel_matrix = np.zeros([self.outDim, self.outDim])
        if (self.pvFlag == 0): ## only pos
            kt_t = self.kernel(a,b)
            for i in range(self.outDum):
                kernel_matrix[i, i] = kt_t
        else:
            dt = 0.001 #determine the smoothness: the bigger the smoother
            ta = a
            tb = b
            tadt = ta + dt
            tbdt = tb + dt

            kt_t = self.kernel(ta,tb)

            kt_dt_temp = self.kernel(ta,tbdt)
            kt_dt = (kt_dt_temp - kt_t) / dt

            kdt_t_temp = self.kernel(tadt,tb)
            kdt_t = (kdt_t_temp - kt_t) / dt

            kdt_dt_temp = self.kernel(tadt,tbdt)
            kdt_dt = (kdt_dt_temp - kt_dt_temp - kdt_t_temp + kt_t) / (pow(dt, 2))



            halfDim = int(np.round(self.outDim/2))
            for i in range(halfDim):
                kernel_matrix[i, i] = kt_t
                kernel_matrix[i, i+halfDim] = kt_dt
                kernel_matrix[i+halfDim, i] = kdt_t
                kernel_matrix[i+halfDim, i+halfDim] = kdt_dt

        return kernel_matrix

    def kernel(self,a,b):
        ker = np.exp((-1) * self.kh * pow((a - b), 2))

        return ker

    #### estimate matrix K_ma,Y_ma
    ## then, invK and W=inv K_ma * Y_ma are updated
    ########## it is time-consuming ##########################
    def kmp_est_Matrix(self):
        Y_ma = np.zeros([self.len * self.outDim, 1])
        K_ma = np.zeros([self.len * self.outDim, self.len * self.outDim])
        C = np.zeros([self.outDim, self.outDim])

        for i in range(self.len):
            for j in range(self.len):
                temp1 = self.data[i, 0]
                temp2 = self.data[j, 0]

                kernel_extend_temp = self.kernel_extend(temp1, temp2)
                K_ma[i * self.outDim: (i + 1) * self.outDim,
                j * self.outDim: (j + 1) * self.outDim] = kernel_extend_temp
                if (i == j):
                    for k in range(self.outDim):
                        index = self.inDim + (k + 1) * self.outDim
                        C[k, :] = self.data[i, index: index + self.outDim]

                    kernel_matr = kernel_extend_temp + self.lamda * C
                    K_ma[i * self.outDim: (i + 1) * self.outDim,
                    j * self.outDim: (j + 1) * self.outDim] = kernel_matr

            Y_ma[i * self.outDim: (i + 1) * self.outDim, 0] = np.transpose(
                self.data[i, self.inDim: (self.inDim + self.outDim)])
        K_inv = np.linalg.inv(K_ma)


        self.W = np.dot(K_inv, Y_ma)
        # ### Matlab multi
        # self.W = np.loadtxt('/home/jiatao/Documents/cvx-a64/cvx/examples/NMPC_QCQP_solution/numerical optimization_imitation learning/nlp_nao_experiments/K_invY.txt')

    #### kmp prediction #####: query: desired time
    def kmp_prediction(self, query):
        Ks = np.zeros([self.outDim,self.outDim * self.len])

        for i in range(self.len):
            temp1 = self.data[i, 0]
            Ks[:,i*self.outDim:(i+1)*self.outDim] = self.kernel_extend(query, temp1)

        des_mean = np.dot(Ks,self.W)

        return  des_mean[:,0]


    #### via_point insert: point format: [t, mean, variance]
    def kmp_point_insert(self, point):
        num_add = 1

        data_ori = self.data

        for i in range(self.len):
            if (np.abs(point[0,0] - self.data[i,0]) <= 0.0005):
                num_add = 0
                replaceNuM = i
                break

        if (num_add > 0.5):
            data_new = np.zeros([data_ori.shape[0]+1, data_ori.shape[1]])
            data_new[0:self.len,:] = copy.deepcopy(self.data)
            data_new[self.len, :] = point[0,:]
            self.data = copy.deepcopy(data_new)
            self.len = data_ori.shape[0]+1
        else:
            self.data[replaceNuM,:] = point[0,:]














