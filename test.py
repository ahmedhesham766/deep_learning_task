import pandas as pd

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

X_Train = Train_Data[Train_Data.columns[1:6]]
X_Test = Test_Data[Test_Data.columns[1:6]]
Y_Train = Train_Data[Train_Data.columns[0]]
Y_Test = Test_Data[Test_Data.columns[0]]



