import numpy as np
import math

class TOPSIS():
    def __init__(self, matrix, weight, J):
        #J: chieu cua diem danh gia, 0 neu chieu tang la tich cuc, 1 neu chieu tang la tieu cuc
        self.J = J
        self.matrix = matrix
        self.m = matrix.shape[0]
        self.n = matrix.shape[1]
        self.rank = np.zeros(self.m)
        self.pos_vector = np.zeros(self.n)
        self.neg_vector = np.zeros(self.n)
        self.neg_distance = np.zeros(self.m)
        self.pos_distance = np.zeros(self.m)
        if weight is not None:
            self.weight = weight
        else:
            self.weight = None

#chuan hoa ma tran dau vao
    def normalize(self):
        norm_matrix = np.zeros((self.m, self.n))
        for i in range(norm_matrix.shape[0]):
            for j in range(norm_matrix.shape[1]):
                norm_matrix[i, j] = self.matrix[i, j] / float(np.sum(self.matrix[:, j]**2))**0.5
        return norm_matrix

# tinh toan trong so entropy trong truong hop weight == None
    def calculate_entropy(self):
        norm_matrix = self.normalize()
        entropy = np.zeros(self.n)
        for j in range(entropy.shape[0]):
            s = 0.0
            for i in range(self.m):
                s += norm_matrix[i, j]*math.log(norm_matrix[i, j])
            entropy[j] = - float(s / math.log(self.m))
        g = np.ones(entropy.shape[0])
        g = g - entropy
        return entropy, g

# tinh toan trong so entropy trong truong hop weight == None
    def calculate_weight_entropy(self):
        entropy, g = self.calculate_entropy()
        a = g / np.sum(g)
        return a


#tinh toan ma tran voi trong so weight
    def weight_matrix(self):
        norm_matrix = self.normalize()
        V = np.zeros((self.m, self.n))
        a = self.calculate_weight_entropy()
        for i in range(self.m):
            for j in range(self.n):
                if self.weight is None:
                    V[i, j] = a[j] * norm_matrix[i, j]
                else:
                    V[i, j] = self.weight[j] * norm_matrix[i, j]
        return V
#tinh toan vector V+ va V-
    def calculate_pis_nis(self):
        V = self.weight_matrix()
        for i in range(self.n):
            if self.J[i] == 0:
                self.pos_vector[i] = np.max(V[:, i])
                self.neg_vector[i] = np.min(V[:, i])
            elif self.J[i] != 0:
                self.pos_vector[i] = np.min(V[:, i])
                self.neg_vector[i] = np.max(V[:, i])

    def calculate_distance_not_weight(self, alpha=0.5):
        self.calculate_pis_nis()
        V = self.weight_matrix()
        delta_neg = np.zeros((self.m, self.n))
        delta_pos = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                delta_neg[i, j] = math.fabs(self.neg_vector[j] - V[i, j])
                delta_pos[i, j] = math.fabs(self.pos_vector[j] - V[i, j])
        r_neg = np.zeros((self.m, self.n))
        r_pos = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                r_pos[i, j] = (np.min(delta_pos) + alpha*np.max(delta_pos)) / (delta_pos[i, j] + alpha*np.max(delta_pos))
                r_neg[i, j] = (np.min(delta_neg) + alpha*np.max(delta_neg)) / (delta_neg[i, j] + alpha*np.max(delta_neg))
        R_neg = np.zeros(self.m)
        R_pos = np.zeros(self.m)
        for i in range(self.m):
            R_pos[i] = np.sum(r_pos[i, :]) / self.n
            R_neg[i] = np.sum(r_neg[i, :]) / self.n
        return R_pos, R_neg

    def calculate_distance(self):
        self.calculate_pis_nis()
        V = self.weight_matrix()
        for i in range(self.m):
            s_neg = 0.0
            s_pos = 0.0
            for j in range(self.n):
                s_neg += (V[i, j] - self.neg_vector[j])**2
                s_pos += (V[i, j] - self.pos_vector[j])**2
            s_neg = s_neg**0.5
            s_pos = s_pos**0.5
            self.neg_distance[i] = s_neg
            self.pos_distance[i] = s_pos

    def ranking(self, alpha=0.5):
        for i in range(self.m):
            if self.weight is not None:
                self.calculate_distance()
                self.rank[i] = self.neg_distance[i]/(self.neg_distance[i] + self.pos_distance[i])
            else:
                R_pos, R_neg = self.calculate_distance_not_weight(alpha)
                self.rank[i] = R_pos[i] / (R_pos[i] + R_neg[i])
    def run(self):
        self.ranking()
        top = np.argmax(self.rank)
        down = np.argmin(self.rank)
        print("Best: %d" % (top + 1))
        print("Worst: %d" % (down + 1))
if __name__ == '__main__':
    matrix = np.array([[1, 2, 2, 1], [2, 3, 1, 4], [4, 4, 4, 4], [2, 2, 2, 3], [1, 1, 1, 1]])
    weight = np.array([0.5, 0.2, 0.2, 0.1])
    topsis = TOPSIS(matrix, None, np.array([0, 0, 0, 0]))
    topsis.run()