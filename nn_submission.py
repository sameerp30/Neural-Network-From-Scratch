import sys
import os
import numpy as np
import pandas as pd
import pickle

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90


class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        '''
        Initialize the neural network.
        Create weights and biases.

        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.


        Parameters
        ----------
                num_layers : Number of HIDDEN layers.
                num_units : Number of units in each Hidden layer.
        '''
        
        self.biases = []
        self.weights = []
        self.num_layers = num_layers
        self.num_units = num_units
        
        for i in range(num_layers):
            if i == 0:
                # Input layer
                self.weights.append(
                    np.random.uniform(-1, 1, size=(NUM_FEATS, num_units[i])))
            else:
                # Hidden layer
                self.weights.append(
                    np.random.uniform(-1, 1, size=(num_units[i-1], num_units[i])))

            self.biases.append(np.random.uniform(-1, 1, size=(num_units[i], 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(num_units[-1], 1)))

    def relu(self, M):
        return M * (M > 0)

    def __call__(self, X):
        '''
        Forward propagate the input X through the network,
        and return the output.

        Note that for a classification task, the output layer should
        be a softmax layer. So perform the computations accordingly

        Parameters
        ----------
                X : Input to the network, numpy array of shape m x d
        Returns
        ----------
                y : Output of the network, numpy array of shape m x 1
        '''

        a = X
        h_states = []
        a_states = []
        params = [(w,self.biases[i]) for i,w in enumerate(self.weights)]
        
        for i, (w, b) in enumerate(params):
            if i == 0:
                h_states.append(a)
            else:
                h_states.append(h)
            a_states.append(a)

            h = np.dot(a, w) + b.T

            if i < len(self.weights)-1:
                a = self.relu(h)
            else:
                a = h

        pred = a

        a_states.append(a)
        h_states.append(a)

        self.a_states = a_states
        self.h_states = h_states

        return pred

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)

        Parameters
        ----------
                X : Input to the network, numpy array of shape m x d
                y : Output of the network, numpy array of shape m x 1
                lamda : Regularization parameter.

        Returns
        ----------
                del_W : derivative of loss w.r.t. all weight values (a list of matrices).
                del_b : derivative of loss w.r.t. all bias values (a list of vectors).

        Hint: You need to do a forward pass before performing backward pass.

        '''

        m = len(y)

        grads = {}
        grads['W'] = {}
        grads['B'] = {}

        length = len(self.weights)

        for i in range(length-1, -1, -1):
            
            if (i == length-1):
                y = np.expand_dims(y, axis=1)
                dA = 1/m*(self.a_states[i+1] - y)
                dZ = dA
            else:
                dA = np.dot(dZ, self.weights[i+1].T)
                dZ = np.multiply(dA, np.int64(self.h_states[i+1] > 0))

            grads['W'][i] = 1/m * \
                np.dot(self.a_states[i].T, dZ) + \
                (lamda)/m * self.weights[i]

            grads['B'][i] = 1/m * np.sum(dZ, axis=0, keepdims=True)
            grads['B'][i] = grads['B'][i].T

        del_W = []
        del_B = []

        for i in range(0, length):
            del_W.append(grads['W'][i])
            del_B.append(grads['B'][i])

        return del_W, del_B


class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate, length):
        '''
        Create a Gradient Descent based optimizer with given
        learning rate.

        Other parameters can also be passed to create different types of
        optimizers.

        Hint: You can use the class members to track various states of the
        optimizer.
        '''
        
        self.learning_rate = learning_rate
        self.alpha = 0.02
        self.beta1 = 0.8
        self.beta2 = 0.999
        self.eps = 1e-8
        self.mw = [0.0 for _ in range(length)]
        self.vw = [0.0 for _ in range(length)]
        self.mb = [0.0 for _ in range(length)]
        self.vb = [0.0 for _ in range(length)]
        self.t = 1

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
                weights: Current weights of the network.
                biases: Current biases of the network.
                delta_weights: Gradients of weights with respect to loss.
                delta_biases: Gradients of biases with respect to loss.
        '''
        size = len(weights)

        updated_W = []
        updated_B = []

        for i in range(0, size):

            self.mw[i] = self.beta1 * self.mw[i] + \
                (1.0 - self.beta1) * delta_weights[i]
            self.mb[i] = self.beta1 * self.mb[i] + \
                (1.0 - self.beta1) * delta_biases[i]

            self.vw[i] = self.beta2 * self.vw[i] + \
                (1.0 - self.beta2) * delta_weights[i]**2
            self.vb[i] = self.beta2 * self.vb[i] + \
                (1.0 - self.beta2) * delta_biases[i]**2

            mwhat = self.mw[i] / (1.0 - self.beta1**(self.t+1))
            vwhat = self.vw[i] / (1.0 - self.beta2**(self.t+1))

            mbhat = self.mb[i] / (1.0 - self.beta1**(self.t+1))
            vbhat = self.vb[i] / (1.0 - self.beta2**(self.t+1))

            updated_W.append(
                weights[i] - self.learning_rate * mwhat / (np.sqrt(vwhat) + self.eps))
            updated_B.append(biases[i] - self.learning_rate *
                             mbhat / (np.sqrt(vbhat) + self.eps))

        self.t += 1

        return updated_W, updated_B


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


def train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target):
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

    m = train_input.shape[0]
    epoch_loss=0.
    patience = 15
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
            weights_updated, biases_updated = optimizer.step(
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
                    with open('./model149.pkl', 'wb') as outp:
                        pickle.dump(net, outp, pickle.HIGHEST_PROTOCOL)
                    
                else: j += 1 
            
            step += 1
        epochs += 1
        
        epoch_loss = epoch_loss*batch_size/m
        print("epoch: {} step: {} train loss: {}".format(epochs, step, epoch_loss))
        

    
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


def read_data():
    '''
    Read the train, dev, and test datasets
    '''
    df_train = pd.read_csv(
        "./data/train.csv")
    df_dev = pd.read_csv(
        "./data/dev.csv")
    df_test = pd.read_csv(
        "./data/test.csv")

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

def evaluate_dev_loss_rmse(net, dev_input, dev_target, batch_size, lamda):
    m = len(dev_input)
    
    epoch_loss = 0
    for i in range(0, m, batch_size):
        batch_input = dev_input[i:i+batch_size]
        batch_target = dev_target[i:i+batch_size]
        pred = net(batch_input)
        
        batch_loss = loss_fn_rmse(batch_target, pred,
                                 net.weights, net.biases, lamda)
        epoch_loss += batch_loss

    return epoch_loss/int(m/batch_size)

def evaluate_dev_loss(net, dev_input, dev_target, batch_size, lamda):
    m = len(dev_input)
    
    epoch_loss = 0
    for i in range(0, m, batch_size):
        batch_input = dev_input[i:i+batch_size]
        batch_target = dev_target[i:i+batch_size]
        pred = net(batch_input)
        
        batch_loss = loss_fn(batch_target, pred,
                                 net.weights, net.biases, lamda)
        epoch_loss += batch_loss

    return epoch_loss/int(m/batch_size)

        
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
    pred = net(inputs)
    return pred

def save_predictions(net, test_input):
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
    
    final_csv = pd.DataFrame(datarows).to_csv("./sample149.csv", header=['Id','Predictions'], index=False)

def main():

    # Hyper-parameters
    max_epochs = 500
    batch_size = 32
    learning_rate = 0.0001
    num_layers = 1
    num_units=[32]
    lamda = 0.1  # Regularization Parameter
    
    train_input, train_target, dev_input, dev_target, test_input = read_data()
    
    params = {}
    train_input = standard_scaler(train_input, params)
    dev_input = standard_scaler(dev_input, params)
    test_input = standard_scaler(test_input, params)
    
    
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate, num_layers+1)
    
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target)
    
    with open('./model149.pkl', 'rb') as inp:
        net = pickle.load(inp)
    
    save_predictions(net, test_input)
    

if __name__ == '__main__':
    main()
