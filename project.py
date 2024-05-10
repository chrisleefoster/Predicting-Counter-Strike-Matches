# CS 7050
# Code for Course Project
# Chris Foster
#
# when the players.csv file is in the same folder, this program will ask for a player's name in
# the dataset, then will train a NN to predict that player's next results. This code will output 
# average error of the NN's prediction
#
# data source
# https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches/data?select=players.csv


import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def readData(fields):
    # Read in columns specified in fields
    data = pd.read_csv('players.csv', usecols=fields)

    # Reverse data so oldest games appear first
    data = data.loc[::-1].reset_index(drop=True)
    return data

def getPlayerData(data):
    # Ask which player's kd we want to predict (case sensitive)
    inp = input("Which player's kd would you like to predict? Example name 'karrigan' (case sensitive)")

    # Check if player is in database
    if  inp in data["player_name"].unique():
        selectedPlayerID = data.loc[data['player_name'] == inp, 'player_id'].iloc[0]
    else:
        raise ValueError("player not in database, please try again")
    
    # this makes a dataframe of the matches the selected player played in
    selectedPlayerData = data.loc[data["player_id"] == selectedPlayerID]

    return selectedPlayerID, selectedPlayerData

def addFeatureLabel(data):
    # sort dataframe by playerID
    sortedData = data.sort_values(by=['player_id'])

    # introduce new column for feature index
    sortedData['featureIndex'] = 0
    
    # assign each player a unique feature index
    playerFeatureIndex = 1
    prevPlayer = sortedData['player_id'].iloc[0]
    for index, row in sortedData.iterrows():
        currentPlayer  = row["player_id"]
        if currentPlayer == prevPlayer:
            sortedData.loc[index, 'featureIndex'] = playerFeatureIndex
        else:
            playerFeatureIndex = playerFeatureIndex + 1
            sortedData.loc[index, 'featureIndex'] = playerFeatureIndex
            prevPlayer = currentPlayer
    return sortedData.sort_index()

def getFeatures(data, selectedPlayerID, selectedPlayerData):
    numFeatures = len(data["player_id"].unique())
    numMatches = len(data.loc[data['player_id'] == selectedPlayerID, 'player_id'])
    features = [ [0 for _ in range(numFeatures + 1)] for _ in range(numMatches)]
    features = np.array(features)   
    i = -1
    prevMatch = 0
    # for each entry in the dataframe
    for index, row in data.iterrows():
        currentMatch  = row["match_id"]
        # check if the match id of the current entry is one of the matches the selected player played in and make sure it is not the selected player
        if currentMatch in selectedPlayerData["match_id"].unique():
            # check currentMatch vs prevMatch
            if currentMatch != prevMatch:
                i = i + 1
            # set the player id to 1 for the current player
            player = row['featureIndex']
            features[i, player] = 1
            # need to store the selected player's kd from this game somewhere
            if row["player_id"] == selectedPlayerID:
                features[i, 0] = row['m1_kddiff']
        prevMatch = currentMatch
    feature = features[:,1:]
    label = features[:,0]
    label = label.reshape(-1,1)
    return feature, label 

def testNetworks(x_train, x_test, y_train, y_test, activations, neurons, learningRates, epochs):
    # number of features in feature vector
    numFeatures = len(x_train[0,:])
    
    # create array of errors for each try
    errors = [[[[0 for _ in range(len(activations))] for _ in range(len(neurons))] for _ in range(len(learningRates))] for _ in range(len(epochs))]
    errors = np.array(errors)
   
    i = 0
    for activation in activations:
        j = 0
        for neuron in neurons:
            k = 0
            for learningRate in learningRates:
                m = 0
                for epoch in epochs:
                    # build model
                    model = keras.Sequential()
                    # input layer
                    model.add(keras.Input(shape=(numFeatures)))
                    # fully connected dense layer
                    model.add(layers.Dense(neuron, activation=activation))
                    # output layer
                    model.add(layers.Dense(1, activation="linear"))
                    # compile network
                    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learningRate), loss="mse")
                    # train network
                    model.fit(x_train, y_train, batch_size=32, epochs=epoch)
                    # predict kd with trained model
                    y_pred = model.predict(x_test)
                    # calculate MSE
                    errors[m,k,j,i] = mean_squared_error(y_test,y_pred)
                    m = m + 1
                k = k + 1
            j = j + 1
        i = i + 1
    return errors


# Specify which columns from players.csv you want
fields = ['player_name','player_id','match_id','m1_kddiff']

# Read in data
data = readData(fields)

# Get a specified player's data
selectedPlayerID, selectedPlayerData = getPlayerData(data)

# relabel player_ids
data = addFeatureLabel(data)

# construct feature vector
features, labels = getFeatures(data, selectedPlayerID, selectedPlayerData)

# standardize label data
scaler=StandardScaler()
scalerFit=scaler.fit(labels)
labels=scalerFit.transform(labels)

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1)

# test different neural network parameters
'''
activations = ["elu", "sigmoid"]
neurons = [5, 10, 25, 50]
learningRates = [0.1, 0.05, 0.01, 0.005]
epochs = [5, 10, 25, 50]
errors = testNetworks(x_train, x_test, y_train, y_test, activations, neurons, learningRates, epochs)
print(errors)
'''

# build model
model = keras.Sequential()
model.add(keras.Input(shape=(len(x_train[0,:]))))
model.add(layers.Dense(3, activation="sigmoid"))
model.add(layers.Dense(1, activation="linear"))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
model.fit(x_train, y_train, batch_size=32, epochs=5)
                    
# predict kd with trained model
y_pred = model.predict(x_test)

# return labels to original form
y_pred=scalerFit.inverse_transform(y_pred)
y_test=scalerFit.inverse_transform(y_test)

# print average error in prediction
error = mean_squared_error(y_test,y_pred)
print("The NN predicted the player's kd_diff with an average error of +/-")
print(math.sqrt(error))