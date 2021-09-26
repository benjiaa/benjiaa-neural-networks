''' adaline.py
    Azalea Yunus and Benji Andrews
    September 2, 2020
    CS343: Neural Networks
    Project 1: Single Layer Networks
    ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np


class Adaline():
    ''' Single-layer neural network
    Network weights are organized [bias, wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    '''
    def __init__(self):
        # Network weights: Bias is stored in self.wts[0], wt for neuron 1 is at self.wts[1],
        # wt for neuron 2 is at self.wts[2], ...
        self.wts = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = None
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        return self.wts.copy()

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)
        NOTE: bias is the 1st element of self.wts. Wts for input neurons 1, 2, 3, ..., M occupy
        the remaining positions.
        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        reshaped_wts = self.wts[np.newaxis,:]


        net_in = np.sum(features @ reshaped_wts[:,1:].T,axis=1) +  self.wts[0]
        #print("netin shape, should be 272",net_in.shape)
        return net_in

    def activation(self, net_in):
        '''
        Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x
        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]
        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = net_in
        return net_act

    def compute_loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)
        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)
        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        sse = 0.5 * np.sum((y - net_act) ** 2)
        return sse

    def compute_accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)
        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples N,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).
        Returns:
        ----------
        float. The accuracy for each input sample in the epoch. ndarray.
            Expressed as proportions in [0.0, 1.0]
        '''
        acc = y_pred[y == y_pred].size / y.size
        return acc

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.
        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples N,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features N,).
            Gradient with respect to the neuron weights in the input feature layer
        '''
        errors = errors[np.newaxis,:]
        grad_bias = -1 * (np.sum(errors))
        grad_wts = -1 * (np.sum((errors @ features),axis=0))
        

        return grad_bias, grad_wts

    def predict(self, features):
        '''Predicts the class of each test input sample
        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]
        NOTE: Remember to apply the activation function!
        '''
        pre_pred = self.net_input(features)
        post_pred = self.activation(pre_pred)
        pred = post_pred.copy()
        pred[pred <= 0] = -1
        pred[pred > 0] = 1
        return pred

    def fit(self, features, y, n_epochs=1000, lr=0.001, early_stopping=False, loss_tol=0.1):
        ''' Trains the network on the input features for self.n_epochs number of epochs
        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            Classes corresponding to each input sample (coded -1 or +1).
        n_epochs: int.
            Number of epochs to use for training the network
        lr: float.
            Learning rate used in weight updates during training
        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.
        TODO:
        1. Initialize the weights according to a Gaussian distribution centered
            at 0 with standard deviation of 0.01. Remember to initialize the bias in the same way.
        2. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        self.wts = np.random.normal(loc=0, scale=0.01, size=(features.shape[1]+1))
        self.loss_history = []
        self.accuracy_history = []
        for e in range(n_epochs):
            #2a: net input
            e_in = self.net_input(features)
            e_act = self.activation(e_in)
            

            #2b: compute error, loss, accuracy
            e_pred = self.predict(features) 
            err = y - e_act
            self.loss_history.append(self.compute_loss(y, e_act))
            self.accuracy_history.append(self.compute_accuracy(y, e_pred))
            

            #2c: backprop 
            new_bias, new_wts = self.gradient(err, features)
            self.wts = self.wts - np.hstack((new_bias,new_wts))*lr
            

            #early stopping if the loss isn't changing fast enough
            if early_stopping and e>1:
                if self.loss_history[-2] - self.loss_history[-1] < loss_tol:
                    break

        return self.loss_history, self.accuracy_history

class Perceptron(Adaline): 
    def activation(self, net_in):
        net_act = np.copy(net_in)
        net_act[net_act >= 0] = 1
        net_act[net_act < 0] = -1
        return net_act











