''' adaline_multi.py
    Azalea Yunus and Benji Andrews
    September 2, 2020
    CS343: Neural Networks
    Project 1: Single Layer Networks
    extension of adaline for classification problems where classes>2
'''
import numpy as np
import adaline


class OneVsRest():
    ''' 
    One-Vs-Rest multi class prediction with single layer neural net(s)
    '''

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def multiple_classification(self, features, y, lr):
        '''
        runs a neural net for every class, filters out the best net_act values for each class
        and returns the proper predicted classes

        returns:
        ----------
        pred: predicted classes ranging from 0 to num_classes. shape=(N,)
        '''
        total_activations = np.zeros((self.num_classes, y.size))
        for i in range(self.num_classes):
            classwise_y = np.where(y == i, 1, -1)
            net = adaline.Adaline()
            net.fit(features, classwise_y, lr=lr)
            netin = net.net_input(features)
            total_activations[i] = net.activation(netin)
       
        
        return np.argmax(total_activations,axis=0)


            






















