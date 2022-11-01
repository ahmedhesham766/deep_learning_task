
import pandas as pd

from sklearn.model_selection import train_test_split





dataset = pd.read_csv('penguins.csv')
# preprocessing
dataset[dataset.columns[0]] = pd.Categorical(
    dataset[dataset.columns[0]],
    categories=['Adelie', 'Gentoo', 'Chinstrap']
).codes

types = [dataset[0:50].copy(), dataset[50:100].copy(), dataset[100:150].copy()]

del dataset

X_Train = []
Y_Train = []
X_Test = []
Y_Test = []

# preprocessing each one of the species alone and generating train and test data for every specie
for data in types:
    data[data.columns[1]].fillna(inplace=True, value=data[data.columns[1]].mean())
    data[data.columns[2]].fillna(inplace=True, value=data[data.columns[2]].mean())
    data[data.columns[3]].fillna(inplace=True, value=data[data.columns[3]].mean())
    data[data.columns[5]].fillna(inplace=True, value=data[data.columns[5]].mean())
    # gender below the only nominal feature
    data[data.columns[4]].fillna(inplace=True, value='unknown')
    data[data.columns[4]] = pd.Categorical(
        data[data.columns[4]],
        categories=['unknown', 'male', 'female']
    ).codes
    x = data[data.columns[1:6]]
    y = data[data.columns[0]]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    X_Train.append(x_train)
    X_Test.append(x_test)
    Y_Train.append(y_train)
    Y_Test.append(y_test)



# concatenating the train and test data together
X_Train = pd.concat([X_Train[0], X_Train[1], X_Train[2]])
X_Test = pd.concat([X_Test[0], X_Test[1], X_Test[2]])
Y_Train = pd.concat([Y_Train[0], Y_Train[1], Y_Train[2]])
Y_Test = pd.concat([Y_Test[0], Y_Test[1], Y_Test[2]])

# randomly shuffling the train and test data
X_Train = X_Train.sample(frac=1, random_state=1).reset_index()
X_Test = X_Test.sample(frac=1, random_state=1).reset_index()
Y_Train = Y_Train.sample(frac=1, random_state=1).reset_index()
Y_Test = Y_Test.sample(frac=1, random_state=1).reset_index()



# num_training_class = int(len(dataset) / 3)
# train_len = int(0.6 * num_training_class)
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=0)
#
# num_training = X_train.shape[0]  # number of training data
# num_testing = X_test.shape[0]  # number of testing data
# epochs = 200  # number of epochs to iterate
# features = X_train.shape[1]  # number of input neurons
# eta = 0.001  # learning rate
# P = Perceptron(features=features, epochs=epochs, x_train_data=X_train, x_test_data=X_test, y_train_data=Y_train,
#                y_test_data=Y_test, eta=eta, num_training=num_training, num_testing=num_testing)
# #
# P.fit()
