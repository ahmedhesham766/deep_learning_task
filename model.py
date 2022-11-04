import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from enum import Enum

dataset = pd.read_csv('penguins.csv')

dataset[dataset.columns[0]] = pd.Categorical(dataset[dataset.columns[0]],
                                             categories=['Adelie', 'Gentoo', 'Chinstrap']).codes

dataset[dataset.columns[4]].fillna(inplace=True, value='unknown')
dataset[dataset.columns[4]] = pd.Categorical(dataset[dataset.columns[4]],
                                             categories=['unknown', 'male', 'female']).codes


class Species(Enum):
    Adelie = 0
    Gentoo = 1
    Chinstrap = 2


class Features(Enum):
    bill_length_mm = 1
    bill_depth_mm = 2
    flipper_length_mm = 3
    gender = 4
    body_mass_g = 5


############## Preprocessing ##############
def preprocess(features, goals, dataset: pd.DataFrame):
    # print(features[0])
    # print(features[1])
    # print(goals[0])
    # print(goals[1])
    data = {'species': dataset['species'].values,
            features[0].name: dataset[features[0].name].values,
            features[1].name: dataset[features[1].name].values}
    data = pd.DataFrame(data=data)

    y1 = goals[0].value * 50  # the start index of y1
    y2 = goals[1].value * 50  # the start index of y1
    data = pd.concat([data[y1:y1 + 50], data[y2:y2 + 50]])

    del y1, y2

    columns = data.columns.values
    indices = []

    for i in range(len(columns)):
        if columns[i] not in [columns[0], features[0].name, features[1].name]:
            indices.append(i)

    columns = np.delete(columns, indices)
    data = data[columns]

    del columns, indices

    type1: pd.DataFrame = data[0:50].copy()
    type2: pd.DataFrame = data[50:100].copy()

    del data
    # print(type1, '\n-----------------------------\n', type2)

    type1[type1.columns[1]].fillna(inplace=True, value=type1[type1.columns[1]].mean())
    type1[type1.columns[2]].fillna(inplace=True, value=type1[type1.columns[2]].mean())
    type1['species'] = 1

    type2[type2.columns[1]].fillna(inplace=True, value=type2[type2.columns[1]].mean())
    type2[type2.columns[2]].fillna(inplace=True, value=type2[type2.columns[2]].mean())
    type2['species'] = -1

    # print(type1, '\n--------------\n', type2)

    train_data = pd.concat([type1[0:30], type2[0:30]])
    test_data = pd.concat([type1[30:50], type2[30:50]])

    del type1, type2
    # print(train_data, '\n------------\n', test_data)

    train_data = train_data.sample(frac=1, random_state=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=1).reset_index(drop=True)

    # print(train_data, '\n------------\n', test_data)

    y_train: pd.DataFrame = train_data[train_data.columns[0]]
    x_train: pd.DataFrame = train_data[train_data.columns[1:3]]
    y_test: pd.DataFrame = test_data[test_data.columns[0]]
    x_test: pd.DataFrame = test_data[test_data.columns[1:3]]

    del train_data, test_data

    return x_train, y_train, x_test, y_test


class Perceptron:

    # initialize perceptron without bias
    def __init__(self, features, goals, epochs, x_train_data: pd.DataFrame, x_test_data: pd.DataFrame,
                 y_train_data: pd.DataFrame, y_test_data: pd.DataFrame, eta, with_bias):

        self.features = features  # input features
        self.goals = goals  # input classes
        self.epochs = epochs  # number of epochs
        self.eta = eta  # learning rate
        self.weight = np.random.rand(len(features) + 1)
        x0 = pd.DataFrame(np.zeros(len(x_train_data)), columns=['bias'])
        self.x_train_data = pd.concat([x0, x_train_data], axis=1)
        x0 = pd.DataFrame(np.zeros(len(x_test_data)), columns=['bias'])
        self.x_test_data = pd.concat([x0, x_test_data], axis=1)
        del x0

        if with_bias:
            self.x_train_data['bias'] = 1
            self.x_test_data['bias'] = 1

        self.y_train_data = y_train_data
        self.y_test_data = y_test_data

    def activation_func_signum(self, x):
        y = np.transpose(self.weight).dot(x)

        if y < 0:
            return -1
        else:
            return 1

    def train(self):  # learn through the number of training samples
        for j in range(self.epochs):
            for i in range(len(self.x_train_data)):
                # fetch data
                x = self.x_train_data.values[i]
                y = self.activation_func_signum(x)
                t = self.y_train_data.values[i]

                # calculate difference
                loss = t - y

                # new weight = old weight + (np.transpose(x).dot(loss).dot(self.eta))
                self.weight = self.weight + np.transpose(x) * loss * self.eta
            # print('epoch ' + str(j) + ', fails = ' + str(fails))
            # # calculate mean squared error for each epoch
            # print('epoch ' + str(j) + ': MSE = ' + str(fails / len(self.x_train_data)))

        # training_accuracy = 100 - ((self.lost[self.num_training - 1] / self.num_training) * 100)
        # print(f'Total samples trained: {self.num_training}')
        # print(f'Training accuracy: {training_accuracy}%')
        # print(f'Total epochs: {self.epochs}')

    def plot(self):
        c1 = pd.DataFrame(columns=[self.x_test_data.columns])
        c2 = pd.DataFrame(columns=[self.x_test_data.columns])
        min_x, max_x = float('inf'), float('-inf')

        for i in range(len(self.y_test_data)):
            x = self.x_test_data.iloc[i]
            if x[1] < min_x:
                min_x = x[1]
            if x[1] > max_x:
                max_x = x[1]
            if self.y_test_data.values[i] == 1:
                c1.loc[len(c1)] = [x[0], x[1], x[2]]
            elif self.y_test_data.values[i] == -1:
                c2.loc[len(c2)] = [x[0], x[1], x[2]]
            else:
                print('false')

        plt.figure('fig')
        plt.scatter(c1[c1.columns[1]], c1[c1.columns[2]], color='grey')
        plt.scatter(c2[c2.columns[1]], c2[c2.columns[2]], color='orange')
        plt.xlabel(c1.columns.values[1])
        plt.ylabel(c1.columns.values[2])

        # w0 + w1 * x1 + w2 * x2 = 0 , x1 is x, x2 is y
        # x2 = (- w0 - w1 * 13) / w2 -> x1 = 13

        w0, w1, w2 = self.weight[0], self.weight[1], self.weight[2]
        x = [min_x, max_x]
        y = [(-w0 - w1 * min_x) / w2, (-w0 - w1 * max_x) / w2]

        plt.plot(x, y, marker='o', color='purple')
        # plt.plot()
        plt.show()

    def test(self):  # predict and calculate testing accuracy
        length = len(self.x_test_data)
        mse = 0
        accuracy = 0
        for i in range(length):
            # fetch data
            x = self.x_test_data.values[i]
            # activation function to calc y^
            y = self.activation_func_signum(x)
            # fetch the real target
            t = self.y_test_data.values[i]

            if t == y:
                accuracy += 1
            else:
                mse += np.square(t - y)

        # calculate testing accuracy
        mse /= length
        accuracy = (accuracy / length) * 100

        print(f'Testing MSE: {mse}')
        print(f'Testing accuracy: {accuracy:.2f}%')
        print('--------------------------------')
