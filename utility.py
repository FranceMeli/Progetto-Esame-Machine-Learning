import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import OneHotEncoder

from network import *
from functions import *



def read_monk(name, split_size=25):
    # read the file and import it as a pandas dataframe
    monk_dataset = pd.read_csv("./monks/"+name+".train", sep=' ', header=None, skipinitialspace=True)
    # set the last column (for monks it is the id) as the index
    monk_dataset.set_index(monk_dataset.shape[1]-1, inplace=True)
    # the first column rapresents the class [0, 1]
    labels = monk_dataset.pop(0)
    labels = labels.to_numpy()[:, np.newaxis]
    indexes = list(range(len(monk_dataset)))
    # we represent each row of the dataframe as a one hot vector
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float32)

    monk = []
    for x, y in zip(monk_dataset, labels):
        monk.append((x, y))
    random.shuffle(monk)

    # split into training and development set
    n = int( (split_size* len(monk_dataset))/100)
    train = monk[n:]
    val = monk[:n]

    #we do the same for the test but without splitting it
    monk_dataset = pd.read_csv("./monks/"+name+".test", sep=' ', header=None, skipinitialspace=True)
    monk_dataset.set_index(monk_dataset.shape[1]-1, inplace=True)
    monk_dataset = monk_dataset.sample(frac=1)
    labels = monk_dataset.pop(0)
    labels = labels.to_numpy()[:, np.newaxis]
    indexes = list(range(len(monk_dataset)))
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray().astype(np.float32)
    test = []
    for x, y in zip(monk_dataset, labels):
        test.append((x, y))
    random.shuffle(test)

    return train, val, test


def read_cup(split_test_size=25, split_val_size=25):
    # read the file and import it as a pandas dataframe
    id_inputs = pd.read_csv("./dataset/ML-CUP20-TR.csv", sep=',', header=None, index_col=False, skiprows=7, skipinitialspace=True, usecols=range(1, 11))
    target = pd.read_csv("./dataset/ML-CUP20-TR.csv", sep=',', header=None, index_col=False, skiprows=7, skipinitialspace=True, usecols=range(11, 13))

    id_inputs = id_inputs.to_numpy()
    target = target.to_numpy()

    train = []
    val = []
    test = []

    for x, y in zip(id_inputs, target):
        train.append((x, y))
    random.shuffle(train)
    # split into training and internal test set
    #if(split_test):
    k = int((split_test_size * len(train))/100)
    test = train[:k]
    train = train[k:]
    # split into training and validation set
    #if(split_val):
    n = int( (split_val_size * len(train))/100)
    val = train[:n]
    train = train[n:]

    return train, val, test

def read_TS_cup():
    blind_test = pd.read_csv("./dataset/ML-CUP20-TS.csv", sep=',', skipinitialspace=True, skiprows=7, header=None, usecols=range(1, 11))
    blind_test = blind_test.to_numpy()
    return blind_test
# ---------- Reading the training set ----------
name_file = 'cup' ##CAMBIARE QUESTO PER PROVARE SU ALTRI DATI
#training, validation, test = read_monk(name_file, split_size= 0)
training, validation, test = read_cup(split_test_size = 20, split_val_size = 0)
blind_test = read_TS_cup()
print(len(blind_test))
print(len(blind_test[0]))
