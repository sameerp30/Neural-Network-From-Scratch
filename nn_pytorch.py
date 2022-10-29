import sys
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import nltk
import numpy as np
import pandas as pd
import copy
from nltk.corpus import brown
import re
import gensim
import gc
from datasets import Dataset
from torch.utils.data import DataLoader
import os
import random


# The seed will be fixed to 42 for this assigmnet.
np.random.seed(28)
torch.manual_seed(28)

NUM_FEATS = 90


class Net(nn.Module):
    
    def __init__(self, parameters):
        super(Net, self).__init__()     
        self.fc1 = nn.Linear(parameters["input_size"], parameters["hidden_size1"])
        torch.nn.init.uniform_(self.fc1.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.fc1.bias, a=-1.0, b=1.0)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(parameters["hidden_size1"], parameters["hidden_size2"])
        torch.nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.fc2.bias, a=-1.0, b=1.0)
        
        self.fc3 = nn.Linear(parameters["hidden_size2"], parameters["num_classes"])
        torch.nn.init.uniform_(self.fc3.weight, a=-1.0, b=1.0)
        torch.nn.init.uniform_(self.fc3.bias, a=-1.0, b=1.0)
        #self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out



def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
            y : targets, numpy array of shape m x 1
            y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
            MSE loss between y and y_hat.
    '''

    cost = 1/(2*len(y)) * np.sum(np.square(y - y_hat))
    return cost
    #raise NotImplementedError


def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.

    Parameters
    ----------
            weights and biases of the network.

    Returns
    ----------
            l2 regularization loss 
    '''
    weights_sum = []
    
    for weight in weights:
        weights_sum.append(np.sum(np.square(weight)))

    return np.sum(weights_sum)/2

    #raise NotImplementedError


def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
            y : targets, numpy array of shape m x 1
            y_hat : predictions, numpy array of shape m x 1
            weights and biases of the network
            lamda: Regularization parameter

    Returns
    ----------
            l2 regularization loss 
    '''

    y = np.expand_dims(y, axis=1)
    
    cost = loss_mse(y, y_hat) + (lamda/len(y)) * \
        loss_regularization(weights, biases)

    return cost

    #raise NotImplementedError
    
def loss_fn_rmse(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

    Parameters
    ----------
            y : targets, numpy array of shape m x 1
            y_hat : predictions, numpy array of shape m x 1
            weights and biases of the network
            lamda: Regularization parameter

    Returns
    ----------
            l2 regularization loss 
    '''

    y = np.expand_dims(y, axis=1)

    cost = rmse(y, y_hat) + (lamda/len(y)) * \
        loss_regularization(weights, biases)

    return cost


def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

    Parameters
    ----------
            y : targets, numpy array of shape m x 1
            y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
            RMSE between y and y_hat.
    '''
    cost = 1/(2*len(y)) * np.sum(np.square(y - y_hat))
    return np.sqrt(cost)
    #raise NotImplementedError


def cross_entropy_loss(y, y_hat):
    '''
    Compute cross entropy loss

    Parameters
    ----------
            y : targets, numpy array of shape m x 1
            y_hat : predictions, numpy array of shape m x 1

    Returns
    ----------
            cross entropy loss
    '''
    #raise NotImplementedError

def train_early(net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target, patience, uid, optimize):
    m = train_input.shape[0]
    epoch_loss=0.
    j = 0  # current running dev step
    step = 0  # it increases after pass through one batch
    prev_dev_loss = 10000000
    epochs = 0
    
    while j < patience:
        epoch_loss = 0
        for i in range(0, m, batch_size):
            batch_input = train_input[i : i + batch_size]
            batch_target = train_target[i : i + batch_size]
            
            pred = net(batch_input)  # forward pass

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)
            
            # Get updated weights based on current weights and gradients
            if optimize == "adam":
                weights_updated, biases_updated = optimizer.step_adam(
                    net.weights, net.biases, dW, db)
            else:
                weights_updated, biases_updated = optimizer.step_sgd(
                    net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred,
                                 net.weights, net.biases, lamda)
            epoch_loss += batch_loss
            
            if step % 300 == 0:  # check dev loss and save model weights
                dev_loss_rmse = evaluate_dev_loss_rmse(net, dev_input, dev_target, batch_size, lamda)
                dev_loss= evaluate_dev_loss(net, dev_input, dev_target, batch_size, lamda)
                print("step: {} dev loss rmse: {} dev loss: {}".format(step, dev_loss_rmse, dev_loss))
                if dev_loss < prev_dev_loss:
                    j = 0
                    prev_dev_loss = dev_loss
                    # save model weights
                    print("Saving model weights.....")
                    with open('./models/model{}.pkl'.format(uid), 'wb') as outp:
                        pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)
                    
                else: j += 1 
            
            step += 1
        epochs += 1
        
        epoch_loss = epoch_loss*batch_size/m
        print("epoch: {} step: {} train loss: {}".format(epochs, step, epoch_loss))

def train_epochs(net, optimizer, criterion, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target, patience, uid, optimize, device):
    
    m = train_input.shape[0]
    epoch_loss=0.
    step = 0  # it increases after pass through one batch
    prev_dev_loss = 10000000
    
    for epochs in range(max_epochs):
        
        epoch_loss = 0
        for i in range(0, m, batch_size):
            batch_input = train_input[i : i + batch_size]
            batch_target = train_target[i : i + batch_size]
            
            net.train()
            X_tensor = torch.FloatTensor(batch_input)
            X_tensor = X_tensor.to(device)

            Y_tensor = torch.FloatTensor(batch_target)
            Y_tensor = Y_tensor.to(device).unsqueeze(axis=1)

            outputs = net(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            

            # Compute loss for the batch
            batch_loss = loss.item()
            epoch_loss += batch_loss
            
            if step % 300 == 0:  # check dev loss and save model weights
                dev_loss_rmse = evaluate_dev_loss_rmse(net, dev_input, dev_target, batch_size, criterion, device)
                dev_loss= evaluate_dev_loss(net, dev_input, dev_target, batch_size, criterion, device)
                print("step: {} dev loss rmse: {} dev loss: {}".format(step, dev_loss_rmse, dev_loss))
                
                if dev_loss < prev_dev_loss:
                    prev_dev_loss = dev_loss
                    # save model weights
                    print("Saving model weights.....")
                    torch.save(net.state_dict(), "./models/nn_model{}.pt".format(uid))
                    torch.save(optimizer.state_dict(), "./models/optimizer{}.pt".format(uid))
            
            step += 1
        
        epoch_loss = epoch_loss*batch_size/m
        print("epoch: {} step: {} train loss: {}".format(epochs, step, epoch_loss))

def train(
        net, optimizer, criterion, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target, patience, uid, optimize, early_stop, device):
    '''
    In this function, you will perform following steps:
            1. Run gradient descent algorithm for `max_epochs` epochs.
            2. For each bach of the training data
                    1.1 Compute gradients
                    1.2 Update weights and biases using step() of optimizer.
            3. Compute RMSE on dev data after running `max_epochs` epochs.

    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    '''

    if early_stop:
        train_early(net, optimizer, criterion, batch_size, max_epochs, train_input, train_target,
        dev_input, dev_target, patience, uid, optimize, device)
    else:
        train_epochs(net, optimizer, criterion, batch_size, max_epochs, train_input, train_target,
        dev_input, dev_target, patience, uid, optimize, device)
        

    
def compute_acc(net, data_input, data_target, batch_size):
    m = len(data_input)
    outputs = None
    
    for i in range(0, m, batch_size):
        batch_input = data_input[i:i+batch_size]
        batch_target = data_target[i:i+batch_size]
        pred = net(batch_input)
        if outputs is None: outputs = pred
        else: outputs = np.append(outputs, pred)
        
    correct = 0
    for i in range(len(outputs)):
        #print(outputs[i], data_target[i])
        if round(outputs[i]) == data_target[i]: correct += 1
    
    return 100*correct/len(data_target)
    
    
def get_test_data_predictions(net, inputs):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.

    Parameters
    ----------
            net : trained neural network
            inputs : test input, numpy array of shape m x d

    Returns
    ----------
            predictions (optional): Predictions obtained from forward pass
                                                            on test data, numpy array of shape m x 1
    '''
    raise NotImplementedError


def read_data(train_path, dev_path, test_path):
    '''
    Read the train, dev, and test datasets
    '''
    df_train = pd.read_csv(
        train_path)
    df_dev = pd.read_csv(
        dev_path)
    df_test = pd.read_csv(
        test_path)

    train_x = df_train.iloc[:, 1:92]
    train_y = df_train['1']
    
    dev_x = df_dev.iloc[:, 1:92]
    dev_y = df_dev['1']
    
    return train_x, train_y, dev_x, dev_y, df_test

def standard_scaler(data, params):
    columns = data.columns
    
    for column in columns:
        if column not in params:
            mean = data[column].mean()
            std = data[column].std()
            params[column] = {}
            params[column]["mean"] = mean
            params[column]["std"] = std
        
        data[column] -= params[column]["mean"]
        data[column] /= params[column]["std"]
    
    return data

def evaluate_dev_loss_rmse(net, dev_input, dev_target, batch_size, criterion, device):
    m = len(dev_input)
    net.eval()
    
    epoch_loss = 0
    for i in range(0, m, batch_size):
        batch_input = dev_input[i:i+batch_size]
        batch_target = dev_target[i:i+batch_size]
        
        X_tensor = torch.FloatTensor(batch_input)
        X_tensor = X_tensor.to(device)

        Y_tensor = torch.FloatTensor(batch_target)
        Y_tensor = Y_tensor.to(device).unsqueeze(axis=1)
        outputs = net(X_tensor)
        loss = criterion(outputs, Y_tensor)
        
        batch_loss = loss.item()
        epoch_loss += batch_loss

    return epoch_loss/int(m/batch_size)

def evaluate_dev_loss(net, dev_input, dev_target, batch_size, criterion, device):
    m = len(dev_input)
    net.eval()
    
    epoch_loss = 0
    for i in range(0, m, batch_size):
        batch_input = dev_input[i:i+batch_size]
        batch_target = dev_target[i:i+batch_size]

        X_tensor = torch.FloatTensor(batch_input)
        X_tensor = X_tensor.to(device)

        Y_tensor = torch.FloatTensor(batch_target)
        Y_tensor = Y_tensor.to(device).unsqueeze(axis=1)
        outputs = net(X_tensor)
        loss = criterion(outputs, Y_tensor)
        
        batch_loss = loss.item()
        
        epoch_loss += batch_loss

    return epoch_loss/int(m/batch_size)

        
def get_test_data_predictions(net, inputs, device):
    '''
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.

    Parameters
    ----------
            net : trained neural network
            inputs : test input, numpy array of shape m x d

    Returns
    ----------
            predictions (optional): Predictions obtained from forward pass
                                                            on test data, numpy array of shape m x 1
    '''
    X_tensor = torch.FloatTensor(inputs)
    X_tensor = X_tensor.to(device)

    outputs = net(X_tensor)
    return outputs.detach().cpu().numpy()

def get_dev_data_predictions(net, inputs):
    pred = net(inputs)
    return pred

def save_predictions_for_analysis(net, dev_input, dev_target, uid):
    prediction = get_dev_data_predictions(net, dev_input)
    int_prediction = []
    
    for i in range(len(prediction)):
        int_prediction.append(float(prediction[i])) 
    
    index=[]
    for i in range(len(int_prediction)):
        index.append(i+1)
    
    datarows=[]
    for i in range(len(int_prediction)):
        row = []
        row.append(index[i])
        row.append(int_prediction[i])
        row.append(dev_target[i])
        datarows.append(row)
    
    final_csv = pd.DataFrame(datarows).to_csv("./models/sample_analysis{}.csv".format(uid), header=['Id','Predictions','labels'], index=False)


def save_predictions(net, test_input, uid):
    prediction = get_test_data_predictions(net, test_input)
    int_prediction = []
    
    for i in range(len(prediction)):
        int_prediction.append(float(prediction[i])) 
    
    index=[]
    for i in range(len(int_prediction)):
        index.append(i+1)
    
    datarows=[]
    for i in range(len(int_prediction)):
        row = []
        row.append(index[i])
        row.append(int_prediction[i])
        datarows.append(row)
    
    final_csv = pd.DataFrame(datarows).to_csv("./models/sample{}.csv".format(uid), header=['Id','Predictions'], index=False)

def feature_sel_corr_matrix(train_input, train_target, dev_input, test_input):
    
    train_ip = pd.concat([train_target,train_input], axis = 1)
    correlation_matrix = train_ip.corr()
    correlated_features = set()
    
    for i in range(len(correlation_matrix .columns)):
        if abs(correlation_matrix.iloc[0, i]) <0.1:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

            
    len_corr_features=len(correlated_features)
    
    print(correlated_features)
    
    train_input.drop(labels=correlated_features, axis=1, inplace=True)
    dev_input.drop(labels=correlated_features, axis=1, inplace=True)
    test_input.drop(labels=correlated_features, axis=1, inplace=True)
    
    return train_input,dev_input,test_input,len_corr_features    


def main():

    # Hyper-parameters
    global NUM_FEATS
    max_epochs = 1000
    batch_size = 64
    uid = 1001
    patience = 32
    optimize = "adam"
    
    train_path = "./data/train_mapping_label.csv"
    dev_path = "./data/dev_mapping_label.csv"
    test_path = "./data/test.csv"
    
    train_input, train_target, dev_input, dev_target, test_input = read_data(train_path, dev_path, test_path)
    #from sklearn.decomposition import PCA
    
    train_input, dev_input, test_input, len_corr_features = feature_sel_corr_matrix(train_input, train_target, dev_input, test_input)
    NUM_FEATS = NUM_FEATS -(len_corr_features)
    print(NUM_FEATS)
    print("********************************")
    
    
    params = {}
    train_input = standard_scaler(train_input, params).to_numpy()
    dev_input = standard_scaler(dev_input, params).to_numpy()
    test_input = standard_scaler(test_input, params).to_numpy()
    
    dev_target = dev_target.to_numpy()
    train_target = train_target.to_numpy()
    
    print(train_input.shape)
    
    parameters = {"input_size": NUM_FEATS, "hidden_size1": 16, "hidden_size2": 16, "num_classes": 1, "learning_rate": 0.001}
    net = Net(parameters)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=parameters["learning_rate"], weight_decay=1e-2, betas=(0.8, 0.999))
    device = torch.device("cpu")
    net.to(device)
    
    train(
        net, optimizer, criterion, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target, patience, uid, optimize, False, device)
    
    #save_predictions(net, test_input, uid)
    

if __name__ == '__main__':
    main()
