import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import copy

class cMeans(object):
    def __init__(self,data,centers,weight):
        self.centers = centers #列が各データで、行が各センターを表す
        self.pre_centers = centers # 評価関数で比較するために前のcentersをとっておく
        self.new_centers = np.zeros(self.centers.shape) #更新するためのcenters
        self.numC = self.centers.shape[1] #centerの列の数を表す
        self.data = data # 要素* データ数(列ベクトル)
        self.numD = self.data.shape[1] #dataの列にデータの個数が来る
        self.U = np.random.dirichlet(np.ones(self.numC),size=self.numD).T # numC行, numD列 縦にsumすると1.0である
        self.pre_U = copy.deepcopy(self.U)
        self.m = weight
        self.diff_U = 10000
        
        # update self.U
        self._members()

    # calculate means
    def means(self):
        self.pre_centers = copy.deepcopy(self.new_centers) # update previous new_centers
        for i in range(self.numC):
            upper_sum = []
            under_sum = []
            for k in range(self.numD):
                upper_sum.append((self.U[i][k] **self.m)* self.data[:,k])
                under_sum.append(self.U[i][k] **self.m)
            result = (sum(upper_sum))/ (sum(under_sum))
            self.new_centers[:,i] = result #new_centersのi列にi番目のclusterセンターを代入
        return None
                
                
    # membership 
    def _members(self):
        self.pre_U = copy.deepcopy(self.U) # update previous self.U
        for k in range(self.numD):
            for i in range(self.numC):
                sum_tmp_0 = []
                for j in range(self.numC):
                    under = self.dist(self.data[:,k],self.centers[:,j]) #d_jk
                    upper = self.dist(self.data[:,k],self.centers[:,i]) #d_ik... j_for文内で定数
                    tmp_0 = (upper/under) ** (2/(self.m-1))
                    sum_tmp_0.append(tmp_0)
                result = (sum(sum_tmp_0)) ** (-1)
                self.U[i][k] = result
        return None
    
    
    def members(self):
        self.pre_U = copy.deepcopy(self.U) # update previous self.U
        for k in range(self.numD):
            for i in range(self.numC):
                sum_tmp_0 = []
                for j in range(self.numC):
                    under = self.dist(self.data[:,k],self.new_centers[:,j]) #d_jk
                    upper = self.dist(self.data[:,k],self.new_centers[:,i]) #d_ik... j_for文内で定数
                    tmp_0 = (upper/under) ** (2/(self.m-1))
                    sum_tmp_0.append(tmp_0)
                result = (sum(sum_tmp_0)) ** (-1)
                self.U[i][k] = result
        return None

    
    # 引き数は計算前のmatrix
    def dist(self,y,v):
        # return norm
#         return pdist([y,v],'minkowski', 2)
        return pdist([y,v],'minkowski', 1)
    

    def estimate(self):
        diff_U = np.absolute(self.U - self.pre_U)
        diff_U = diff_U.sum()
        self.diff_U = diff_U
        return None
    
    def new_member(self,new_array):
        ar_result = []
        for i in range(self.numC):
            sum_tmp_0 = []
            for j in range(self.numC):
                under = self.dist(new_array,self.centers[:,j]) #d_jk
                upper = self.dist(new_array,self.centers[:,i]) #d_ik... j_for文内で定数
                tmp_0 = (upper/under) ** (2/(self.m-1))
                sum_tmp_0.append(tmp_0)
            result = (sum(sum_tmp_0)) ** (-1)
            ar_result.append(result)
        ar_result = np.array(ar_result)
        return ar_result
    
    
    def fit(self):
        i= 0
        while self.diff_U > 0.001:
            self.means()
            self.members()
            self.estimate()
            i += 1
            print("{}-th iteration".format(i))
            print("U difference is {}".format(self.diff_U))
    