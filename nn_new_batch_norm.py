import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 78


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
        self.gamma = []
        self.beta = []
        self.num_layers = num_layers
        self.num_units = num_units
        for i in range(num_layers):
            if i == 0:
                # Input layer
                self.weights.append(
                    np.random.uniform(-1, 1, size=(NUM_FEATS, num_units)))
            else:
                # Hidden layer
                self.weights.append(
                    np.random.uniform(-1, 1, size=(num_units, num_units)))
                
                self.gamma.append(np.ones((num_units,1)))
                self.beta.append(np.zeros((num_units,1)))

            self.biases.append(np.random.uniform(-1, 1, size=(num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(num_units, 1)))
        

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
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if i == 0:
                h_states.append(a)
            else:
                h_states.append(h)
            a_states.append(a)
            h = np.dot(a, w) + b.T
            eps=1e-05 
            momentum = 0.01
            
            if i < len(self.weights)-1:
                if(i!=0):
                    h = batch_norm_train(h, self.gamma[i-2], self.beta[i-2], eps, momentum)
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
        grads['gamma'] = {}
        grads['beta'] = {}

        length = len(self.weights)

        for i in range(length-1, -1, -1):
            if (i == length-1):
                y = np.expand_dims(y, axis=1)
                
                dA = 1/m*(self.a_states[i+1] - y)
                #dA = 1/m*(self.a_states[i+1] - y)*loss**(-1/2)
                dZ = dA 
            else:
                dA = np.dot(dZ, self.weights[i+1].T)
                dZ_bar = np.multiply(dA, np.int64(self.h_states[i+1] > 0))
                
                if(i != 0):
                    dZ = np.multiply(dZ_bar, self.gamma[i-2].T)
                else: dZ = dZ_bar
            
            grads['W'][i] = 1/m * \
                np.dot(self.a_states[i].T, dZ) + \
                (lamda)*self.weights[i]
            grads['B'][i] = 1/m * np.sum(dZ, axis=0, keepdims=True)
            grads['B'][i] = grads['B'][i].T
            
            if i != 0 and i != length - 1:
                grads['beta'][i] = 1/m * np.sum(dZ_bar.T , axis=1, keepdims=True)
                grads['gamma'][i] = 1/m * np.sum(self.h_states[i+1].T , axis=1, keepdims=True)

        del_W = []
        del_B = []
        del_gamma = []
        del_beta = []

        for i in range(0, length):
            del_W.append(grads['W'][i])
            del_B.append(grads['B'][i])
        for i in range(1, length-1):
            del_beta.append(grads['gamma'][i])
            del_gamma.append(grads['beta'][i])

        return del_W, del_B , del_beta , del_gamma


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
        self.mg = [0.0 for _ in range(length-2)]
        self.vg = [0.0 for _ in range(length-2)]
        self.mbeta = [0.0 for _ in range(length-2)]
        self.vbeta = [0.0 for _ in range(length-2)]
        
        self.t = 1

        #raise NotImplementedError

    def step(self, weights, biases, gamma, beta ,delta_weights, delta_biases , delta_gamma, delta_beta):
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
        updated_gamma = []
        updated_beta = []
        
        
        for i in range(0, size):

            self.mw[i] = self.beta1 * self.mw[i] + \
                (1.0 - self.beta1) * delta_weights[i]
            self.mb[i] = self.beta1 * self.mb[i] + \
                (1.0 - self.beta1) * delta_biases[i]

            self.vw[i] = self.beta2 * self.vw[i] + \
                (1.0 - self.beta2) * delta_weights[i]**2
            self.vb[i] = self.beta2 * self.vb[i] + \
                (1.0 - self.beta2) * delta_biases[i]**2
            
            if(i!=0 and i!=size-1):
                self.mg[i-2] = self.beta2 * self.mg[i-2] + \
                    (1.0 - self.beta2) * delta_gamma[i-2]**2
                self.mbeta[i-2] = self.beta2 * self.mbeta[i-2] + \
                    (1.0 - self.beta2) * delta_beta[i-2]**2
            
                self.vg[i-2] = self.beta2 * self.vg[i-2] + \
                    (1.0 - self.beta2) * delta_gamma[i-2]**2
                self.vbeta[i-2] = self.beta2 * self.vbeta[i-2] + \
                    (1.0 - self.beta2) * delta_beta[i-2]**2
            

            mwhat = self.mw[i] / (1.0 - self.beta1**(self.t+1))
            vwhat = self.vw[i] / (1.0 - self.beta2**(self.t+1))

            mbhat = self.mb[i] / (1.0 - self.beta1**(self.t+1))
            vbhat = self.vb[i] / (1.0 - self.beta2**(self.t+1))
            
            if(i!=0 and i!=size-1):
                mghat = self.mg[i-2] / (1.0 - self.beta1**(self.t+1))
                vghat = self.vg[i-2] / (1.0 - self.beta1**(self.t+1))
            
                mbetahat = self.mbeta[i-2] / (1.0 - self.beta1**(self.t+1))
                vbetahat = self.vbeta[i-2] / (1.0 - self.beta1**(self.t+1))
            
     

            updated_W.append(
                weights[i] - self.learning_rate * mwhat / (np.sqrt(vwhat) + self.eps))
            updated_B.append(biases[i] - self.learning_rate *
                             mbhat / (np.sqrt(vbhat) + self.eps))
            
            
            if(i!=0 and i!=size-1):
                updated_gamma.append(
                    gamma[i-2] - self.learning_rate * mghat / (np.sqrt(vghat) + self.eps))
                updated_beta.append(beta[i-2] - self.learning_rate *mbetahat / (np.sqrt(vbetahat) + self.eps))
        
        
        self.t += 1

        return updated_W, updated_B , updated_gamma , updated_beta

        #raise NotImplementedError
        

def batch_norm_train(X, gamma, beta, eps, momentum):
    
    mean = X.mean(axis=0)
    
    var = ((X - mean) ** 2).mean(axis=0)
    
    X_hat = (X - mean) / np.sqrt(var + eps)
    # Update the mean and variance using moving average
    #moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
    #moving_var = (1.0 - momentum) * moving_var + momentum * var
    #print(gamma.T.shape , X_hat.shape , beta.T.shape)
    Y = np.multiply(gamma.T, X_hat) + beta.T  # Scale and shift
    
    return Y


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
    #cost = rmse(y, y_hat) + lamda * \
    #    loss_regularization(weights, biases)
    cost = loss_mse(y, y_hat) + lamda * \
        loss_regularization(weights, biases)

    return cost

    #raise NotImplementedError


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
        dev_input, dev_target
):
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

    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            pred = net(batch_input)

            # Compute gradients of loss w.r.t. weights and biases
            dW, db, dgamma, dbeta = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated , updated_gamma , updated_beta  = optimizer.step(
                net.weights, net.biases, net.gamma , net.beta , dW, db, dgamma, dbeta)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated
            net.gamma = updated_gamma
            net.beta = updated_beta

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred,
                                 net.weights, net.biases, lamda)
            
            epoch_loss += batch_loss

            #print(e, i, rmse(batch_target, pred), batch_loss)
        total_dev_loss=get_dev_data_predictions(net,dev_input,dev_target)
        print(e, epoch_loss/int(m/batch_size), total_dev_loss)
        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    #dev_pred = net(dev_input)
    #dev_rmse = rmse(dev_target, dev_pred)

    #print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_dev_data_predictions(net, inputs, outputs):
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
    
    dev_size = inputs.shape[0]
    total_dev_loss=0
    batch_size=100
    for i in range(0, dev_size, batch_size):
            dev_batch_input = inputs[i:i+batch_size]
            dev_batch_target = outputs[i:i+batch_size]
            pred = net(dev_batch_input)
            dev_batch_loss=loss_fn(dev_batch_target, pred,net.weights, net.biases, 0.01)
            total_dev_loss+=dev_batch_loss
    total_dev_loss=int(total_dev_loss*batch_size/dev_size)
    return total_dev_loss
    #raise NotImplementedError


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


def main():

    global NUM_FEATS
    NUM_FEATS = 78
        
    # Hyper-parameters
    max_epochs = 500
    batch_size = 32
    learning_rate = 0.001
    num_layers = 1
    num_units = 128
    lamda = 0.01  # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()

    scaler = MinMaxScaler()
    columns = train_input.columns
    
    train_input[columns] = scaler.fit_transform(train_input[columns])
    dev_input[columns] = scaler.fit_transform(dev_input[columns])
    corr = ["59", "25", "22", "19", "24", "21", "17"]
    train_input = train_input.drop(corr, axis=1)
    dev_input = dev_input.drop(corr, axis=1)
    
    l1_based = ['56', '55', '74', '81', '82']

    train_input = train_input.drop(l1_based, axis=1)
    dev_input = dev_input.drop(l1_based, axis=1)
    
    train_input = train_input.to_numpy()
    dev_input = dev_input.to_numpy()

    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate, num_layers+1)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    


if __name__ == '__main__':
    main()
