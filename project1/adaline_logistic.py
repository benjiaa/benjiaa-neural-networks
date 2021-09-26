''' adaline_logistic.py
    Azalea Yunus and Benji Andrews
    September 2, 2020
    CS343: Neural Networks
    Project 1: Single Layer Networks
    ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np
import adaline


class AdalineLogistic(adaline.Adaline):
    ''' Single-layer neural network
    Network weights are organized [bias, wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    '''
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
        # sigmoid activation function        
        net_act = 1 / (1 + np.exp(-net_in))
        return net_act

    def compute_loss(self, y, net_act):
        ''' Computes the cross-entropy loss (over a single training epoch)
        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)
        Returns:
        ----------
        float. The cross-entropy loss (across a single training epoch).
        '''
        cross_entropy = np.sum((-y@np.log(net_act)-(1-y)) @ np.log(1-net_act))

        return -cross_entropy 

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
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        return pred
    
    def predict_probability(self, features):
        '''Predicts the class of each test input sample
        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        Returns:
        ----------
        pred: ndarray. predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]
        NOTE: Remember to apply the activation function!
        cert: ndarray. The precent certaint the prediction is in each class. Shape = [Num samples N,]
        '''
        pre_pred = self.net_input(features)
        post_pred = self.activation(pre_pred)
        pred = post_pred.copy()
        cert = post_pred.copy()

        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        cert = np.where(pred==0,np.abs(cert-1),cert)
        
        return pred, cert

