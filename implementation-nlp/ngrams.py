
import numpy as np
from sklearn.model_selection import train_test_split
import os
import math
from ngrams import utils


class Classifier():
    def __init__(self):
        self.uni_grams = {}
        self.bi_grams = {}
        self.tri_grams = {}
        self.labels = []
        self.__classes__ = []


    def fit(self, X_train, y_train):
        self.__classes__ = list(set(y_train))
        self.prob_c = np.zeros(len(self.__classes__))
        count = 0
        for i, c in enumerate(y_train):
            sent = X_train[i]
            self.prob_c[int(c)] += 1
            for j in range(len(sent)):
                token_3 = sent[j]
                if j == 0:
                    token_1 = '*'
                    token_2 = '*'
                elif j == 1:
                    token_1 = '*'
                    token_2 = sent[j-1]
                else:
                    token_1 = sent[j-2]
                    token_2 = sent[j-1]
                uni_gram = (
                    token_3,
                    c
                )
                bi_gram = (
                    token_2,
                    token_3,
                    c
                )
                tri_gram = (
                    token_1,
                    token_2,
                    token_3,
                    c
                )
                if uni_gram not in self.uni_grams:
                    self.uni_grams[uni_gram] = 1
                else:
                    self.uni_grams[uni_gram] += 1
                if bi_gram not in self.bi_grams:
                    self.bi_grams[bi_gram] = 1
                else:
                    self.bi_grams[bi_gram] += 1
                if tri_gram not in self.tri_grams:
                    self.tri_grams[tri_gram] = 1
                else:
                    self.tri_grams[tri_gram] += 1
        self.prob_c = float(self.prob_c/len(y_train))
        self.N = len(self.uni_grams) + len(self.bi_grams) + len(self.tri_grams)

    def get_prob(self, token_3, token_2, token_1, c):
        uni_gram = (
            token_3,
            c
        )
        bi_gram = (
            token_2,
            token_3,
            c
        )
        tri_gram = (
            token_1,
            token_2,
            token_3,
            c
        )
        if tri_gram in self.tri_grams:
            prob_tri = float(self.tri_grams[tri_gram] + 1)/float(self.bi_grams[bi_gram])
            prob_bi = float(self.bi_grams[bi_gram]+1)/float(self.uni_grams[uni_gram])
            prob_uni = float(self.uni_grams[uni_gram] + 1)/float(self.N)
        else:
            prob_tri = 0.0
            if bi_gram in self.bi_grams:
                prob_bi = float(self.bi_grams[bi_gram] + 1) / float(self.uni_grams[uni_gram])
                prob_uni = float(self.uni_grams[uni_gram] + 1) / float(self.N)
            else:
                prob_bi = 0.0
                if uni_gram in self.uni_grams:
                    prob_uni = float(self.uni_grams[uni_gram] + 1) / float(self.N)
                else:
                    prob_uni = 1.0/float(self.N)
        lambda_1 = 0.1
        lambda_2 = 0.3
        lambda_3 = 0.7
        prob = lambda_1*prob_uni + lambda_2*prob_bi + lambda_3*prob_tri
        return float(prob)

    def predict(self, X_test):
        y_pred = []
        for i in range(len(X_test)):
            sent = X_test[i]
            _max = None
            _c = None
            for c in self.__classes__:
                _prob = math.log(self.prob_c[int(c)])
                for j in range(len(sent)):
                    token_3 = sent[j]
                    if j == 0:
                        token_2 = '*'
                        token_1 = '*'
                    elif j == 1:
                        token_2 = sent[j-1]
                        token_1 = '*'
                    else:
                        token_2 = sent[j-1]
                        token_1 = sent[j-2]
                    prob_w = self.get_prob(token_3, token_2, token_1, c)
                    _prob += math.log(prob_w)
                if _max == None:
                    _max = _prob
                    _c = c
                else:
                    if _max < _prob:
                        _max = _prob
                        _c = c
            y_pred.append(_c)
        return y_pred



if __name__ == '__main__':
    X, y = utils.parse_file('traning_data.txt')
    for i in range(len(X)):
        X[i] = utils.preprocess(X[i])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    model = Classifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    count = 0
    for i, c in enumerate(y_pred):
        if c == y_test[i]:
            count += 1
    print('Acc {} %'.format(count/len(y_pred)))




