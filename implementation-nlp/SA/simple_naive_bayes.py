import numpy as np
import utils
from sklearn.model_selection import train_test_split
import math

class MultinomialNB():
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.__classes = []
        self.prob = None
        self.prob_c = None
    def fit(self, X_train, y_train):
        self.__classes = list(set(y_train))
        count = np.zeros((len(self.__classes), X_train.shape[1]))
        len_class = np.zeros(len(self.__classes))
        self.prob_c = np.zeros(len(self.__classes))
        self.prob = np.zeros((len(self.__classes), X_train.shape[1]))
        for i, c in enumerate(y_train):
            self.prob_c[int(c) - 1] += 1
            len_class[int(c) - 1] += np.sum(X_train[i])
            count[int(c) - 1] += X_train[i]
        self.prob_c = self.prob_c/X_train.shape[0]
        for c in self.__classes:
            self.prob[int(c) - 1] = (count[int(c) - 1] + self.alpha)/(len_class[int(c) - 1] + self.alpha*X_train.shape[1])


    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            _max = -99999.0
            _c = 0
            for c in self.__classes:
                _prob = math.log(self.prob_c[int(c) - 1])
                for j in range(X_test.shape[1]):
                    if X_test[i][j] != 0:
                        _prob += math.log(self.prob[int(c) - 1][j])*X_test[i][j]
                if _prob > _max:
                    _max = _prob
                    _c = c
            y_pred.append(_c)
        return y_pred

    def accuracy(self, y_test, y_pred):
        count = 0
        for i in range(len(y_test)):
            if str(y_test[i]) == str(y_pred[i]):
                count += 1
        print('Acc: %.2f' % (count*100/len(y_test)), end=' %')


if __name__ == '__main__':
    X_, y = utils.parse_file('training_data.txt')
    utils.build_dict(X_)
    DICT = utils.load_dict()
    X = np.zeros((len(X_), len(DICT)))
    for i in range(len(X_)):
        X[i] = utils.bag_of_word(X_[i], DICT)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    model = MultinomialNB(0.1)
    model.fit(X_train, y_train)
    print(X_test.shape[1])
    y_pred = model.predict(X_test)
    model.accuracy(y_test, y_pred)
