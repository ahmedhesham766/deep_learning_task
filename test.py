import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv('penguins.csv')
# preprocessing
dataset[dataset.columns[0]] = pd.Categorical(
    dataset[dataset.columns[0]],
    categories=['Adelie', 'Gentoo', 'Chinstrap']
).codes

dataset[dataset.columns[1]].fillna(inplace=True, value=dataset[dataset.columns[1]].mean())
dataset[dataset.columns[2]].fillna(inplace=True, value=dataset[dataset.columns[2]].mean())
dataset[dataset.columns[3]].fillna(inplace=True, value=dataset[dataset.columns[3]].mean())
dataset[dataset.columns[5]].fillna(inplace=True, value=dataset[dataset.columns[5]].mean())
# gender below the only nominal feature
dataset[dataset.columns[4]].fillna(inplace=True, value='unknown')
dataset[dataset.columns[4]] = pd.Categorical(
    dataset[dataset.columns[4]],
    categories=['unknown', 'male', 'female']
).codes

Train_Data = pd.concat([dataset[0:30], dataset[50:80], dataset[100:130]])
Test_Data = pd.concat([dataset[30:50], dataset[80:100], dataset[130:150]])

Train_Data = Train_Data.sample(frac=1, random_state=1).reset_index(drop=True)
Test_Data = Test_Data.sample(frac=1, random_state=1).reset_index(drop=True)

X_Train: pd.DataFrame = Train_Data[Train_Data.columns[1:6]]
X_Test: pd.DataFrame = Test_Data[Test_Data.columns[1:6]]
Y_Train: pd.DataFrame = Train_Data[Train_Data.columns[0]]
Y_Test: pd.DataFrame = Test_Data[Test_Data.columns[0]]

del dataset, Train_Data, Test_Data


class Perceptron:

    # initialize perceptron
    def __init__(self, features, epochs, x_train_data: pd.DataFrame, x_test_data: pd.DataFrame,
                 y_train_data: pd.DataFrame, y_test_data: pd.DataFrame, eta):
        self.features = features  # input features
        self.epochs = epochs  # number of epochs
        self.eta = eta  # learning rate
        # self.lost = np.zeros(num_training)  # error difference between predicted value and generated value
        # self.mse = np.zeros(self.epochs)  # mean squared error for plotting the graph
        self.weight = np.zeros(len(features) + 1)  # initial weight

        x0 = pd.DataFrame(np.ones(len(x_train_data)), columns=['bias'])
        self.x_train_data = pd.concat([x0, x_train_data], axis=1)
        x0 = pd.DataFrame(np.ones(len(x_test_data)), columns=['bias'])
        self.x_test_data = pd.concat([x0, x_test_data], axis=1)
        del x0
        self.y_train_data = y_train_data
        self.y_test_data = y_test_data

    def activation_func(self, x):
        y = np.transpose(self.weight).dot(x)

        if y < 0:
            return -1
        else:
            return 1

    def train(self):  # learn through the number of training samples
        for j in range(self.epochs):
            fails = 0
            for i in range(len(self.x_train_data)):

                x = self.x_train_data.values[i]
                y = self.activation_func(x)
                t = self.y_train_data.values[i]

                # calculate difference
                loss = t - y

                if loss == 0:
                    pass
                else:
                    fails += 1
                    self.weight = self.weight + (np.transpose(x).dot(loss).dot(self.eta))
            print('epoch ' + str(j) + ', fails = ' + str(fails))
            # # calculate mean squared error for each epoch
            print('epoch ' + str(j) + ': MSE = ' + str(fails/len(self.x_train_data)))

        # training_accuracy = 100 - ((self.lost[self.num_training - 1] / self.num_training) * 100)
        # print(f'Total samples trained: {self.num_training}')
        # print(f'Training accuracy: {training_accuracy}%')
        # print(f'Total epochs: {self.epochs}')

    def plot(self):
        c1 = pd.DataFrame(columns=[x_train_data.columns])
        c2 = pd.DataFrame(columns=[x_train_data.columns])
        c3 = pd.DataFrame(columns=[x_train_data.columns])

        for i in range(len(self.y_train_data)):
            x = self.x_train_data.iloc[i]
            if self.y_train_data.values[i] == 0:
                c1.loc[len(c1)] = [x[1], x[2]]
            elif self.y_train_data.values[i] == 1:
                c2.loc[len(c2)] = [x[1], x[2]]
            elif self.y_train_data.values[i] == 2:
                c3.loc[len(c3)] = [x[1], x[2]]
            else:
                print('false')
        plt.figure('fig')
        plt.scatter(c1[c1.columns[0]], c1[c1.columns[1]])
        plt.scatter(c2[c2.columns[0]], c2[c2.columns[1]])
        plt.scatter(c3[c3.columns[0]], c3[c3.columns[1]])
        plt.xlabel(feature[0])
        plt.ylabel(feature[1])
        plt.plot()
        plt.show()

    # def predict(self):  # predict and calculate testing accuracy
    #
    #     for i in range(self.num_testing):
    #         # fetch data
    #         x = self.x_test_data[i, 0:self.features]
    #
    #         # activation function
    #         y = self.activation_func(x)
    #
    #         # calculate error points
    #         if y != self.y_test_data[i]:
    #             self.error_points += 1
    #
    #     # calculate testing accuracy
    #     testing_accuracy = 100 - ((self.error_points / self.num_testing) * 100)
    #
    #     print(f'Total samples tested: {self.num_testing}')
    #     print(f'Total error points: {self.error_points}')
    #     print(f'Testing accuracy: {testing_accuracy:.2f}%')


feature = [X_Test.columns[0], X_Test.columns[1]]
x_train_data = X_Train[X_Train.columns[[0, 1]]]
x_test_data = X_Test[X_Test.columns[[0, 1]]]
per = Perceptron(features=feature, x_train_data=x_train_data, x_test_data=x_test_data,
                 y_train_data=Y_Train, y_test_data=Y_Test, eta=0.1, epochs=1000)
per.train()
per.plot()
