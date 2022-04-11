import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification


# 模型里的模块：
# sigmoid 映射到概率的函数
# model 返回预测结果值
# cost 根据参数计算损失
# gradient 计算每个参数的梯度方向
# descent 进行参数更新
# accuracy 计算精度


class LogisticRegression1():
    def __init__(self, learning_rate, n_iterations):
        # 初始化学习率和迭代次数
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def initialize_weights(self, x, y):
        # 初始化参数
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

    def fit(self, x, y):
        self.initialize_weights(x, y)

        # gradient descent
        for i in range(self.n_iterations):
            self.update_weight()

    def update_weight(self):
        # gradient descent
        h = self.sigmoid(self.x.dot(self.w) + self.b)
        tmp = np.reshape(h - self.y.T, self.m)
        dw = np.dot(self.x.T, tmp) / self.m
        db = np.sum(tmp) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, x):
        h = self.sigmoid(x.dot(self.w) + self.b)
        res = np.where(h > 0.5, 1, 0)
        return res


def evaluation_model(Y_pred, Y_test):
    correctly_classified = 0
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1
        count = count + 1

    print("Accuracy on test set :  ", (
            correctly_classified / count) * 100)


def main():
    # load dataset
    df = pd.read_csv("dataset-test/diabetes.csv")
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1:].values

    # Splitting dataset into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

    # load model
    model = LogisticRegression1(learning_rate=0.01, n_iterations=1000)

    # model training
    print(x_train.shape)
    print(y_train.shape)
    model.fit(x_train, y_train)

    # prediction on test set
    y_pred = model.predict(x_test)
    print(y_pred)

    # evaluation
    evaluation_model(y_pred, y_test)


pd.set_option('display.width', None)
main()











