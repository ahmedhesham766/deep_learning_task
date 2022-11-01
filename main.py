import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('penguins.csv')

X = dataset[dataset.columns[1:6]]
Y = dataset[dataset.columns[0]]


class Perceptron:

    # initialize perceptron
    def __init__(self, features, epochs, x_train_data, x_test_data, y_train_data, y_test_data,
                 eta, num_training, num_testing):
        self.features = features  # number of input features
        self.bias = np.random.randn()  # bias
        self.epochs = epochs  # number of epochs
        self.eta = eta  # learning rate
        self.lost = np.zeros(num_training)  # error difference between predicted value and generated value
        self.mse = np.zeros(self.epochs)  # mean squared error for plotting the graph
        self.weight = np.random.rand(features)  # initial weight
        self.num_training = num_training  # number of training samples
        self.num_testing = num_testing  # number of testing samples
        self.error_points = 0  # to keep track of the total testing error

        self.x_train_data = x_train_data
        self.x_test_data = x_test_data

        self.y_train_data = y_train_data
        self.y_test_data = y_test_data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def activation_func(self, x):
        y = np.transpose(self.weight).dot(x) + self.bias

        # sigmoid function
        y = 1 / (1 + np.exp(-y))

        if y == 1:
            return 1
        elif y < 1:
            return 0
        else:
            return 2

    def fit(self):  # learn through the number of training samples

        for j in range(self.epochs):

            for i in range(self.num_training):
                # fetch data
                x = self.x_train_data[i, 1:self.features]

                # fetch desired output from dataset
                t = self.y_train_data[i]

                # activation function
                y = self.activation_func(x)

                # calculate difference
                self.lost[i] = t - y

                # new weight
                new_weight = self.weight + x.dot(self.lost[i] * self.eta)

                # at any point if the weights are similar, then skip to the next epoch
                if y != t:
                    break

                # otherwise, set the new weight as current weight
                self.weight = new_weight

            # calculate mean squared error for each epoch
            self.mse[j] = np.square(self.lost).mean()

        training_accuracy = 100 - ((self.lost[self.num_training - 1] / self.num_training) * 100)
        print(f'Total samples trained: {self.num_training}')
        print(f'Training accuracy: {training_accuracy}%')
        print(f'Total epochs: {self.epochs}')

    def plot_fit(self):
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error (mse)')
        plt.title('Training accuracy')
        plt.plot(self.mse)
        plt.show()

    def predict(self):  # predict and calculate testing accuracy

        for i in range(self.num_testing):
            # fetch data
            x = self.x_test_data[i, 0:self.features]

            # activation function
            y = self.activation_func(x)

            # calculate error points
            if y != self.y_test_data[i]:
                self.error_points += 1

        # calculate testing accuracy
        testing_accuracy = 100 - ((self.error_points / self.num_testing) * 100)

        print(f'Total samples tested: {self.num_testing}')
        print(f'Total error points: {self.error_points}')
        print(f'Testing accuracy: {testing_accuracy:.2f}%')


num_training_class = int(len(dataset) / 3)
train_len = int(0.6 * num_training_class)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=0)
num_training = X_train.shape[0]  # number of training data
num_testing = X_test.shape[0]  # number of testing data
epochs = 200  # number of epochs to iterate
features = X_train.shape[1]  # number of input neurons
eta = 0.001  # learning rate
P = Perceptron(features, epochs, X_train, X_test, Y_train, Y_test, eta, num_training, num_testing)
#
P.fit()
