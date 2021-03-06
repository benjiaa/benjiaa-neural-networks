''' mlp.py
    Constructs, trains, tests 3 layer multilayer layer perceptron networks
    Benji Andrews and Azalea Yunus
    CS343: Neural Networks
    Project 2: Multilayer Perceptrons
'''
import numpy as np


class MLP():
    '''
    MLP is a class for multilayer perceptron network.

    The structure of our MLP will be:

    Input layer (X units) ->
    Hidden layer (Y units) with Rectified Linear activation (ReLu) ->
    Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that
    the current input sample belongs to class i.

    NOTE: We will keep our bias weights separate from our feature weights to simplify computations.
    '''
    def __init__(self, num_input_units, num_hidden_units, num_output_units):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units
        self.initialize_wts(num_input_units, num_hidden_units, num_output_units)

    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y_wts.copy()

    def initialize_wts(self, M, H, C, std=0.1):
        ''' Randomly initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.y_wts, self.y_b and self.z_wts, self.z_b
        with the appropriate size according to the normal distribution with standard deviation
        `std` and mean of 0.
          - For wt shapes, they should be be equal to (#prev layer units, #associated layer units)
            for example: self.y_wts has shape (M, H)
          - For bias shapes, they should equal the number of units in the associated layer.
            for example: self.y_b has shape (H,)
        '''
        # keep the random seed for debugging/test code purposes
        np.random.seed(0)
        # mostly re-used from softmax.py, adjusted for shape of y and z layers
        self.y_wts = np.random.normal(loc=0, scale=std, size=(M, H))
        self.y_b = np.random.normal(loc=0, scale=std, size=(H))
        self.z_wts = np.random.normal(loc=0, scale=std, size=(H, C))
        self.z_b = np.random.normal(loc=0, scale=std, size=(C))

    def accuracy(self, y, y_pred):
        ''' Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        # re-used from softmax.py
        acc = y_pred[y == y_pred].size / y.size
        return acc

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        # re-used from softmax.py
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    def predict(self, features):
        ''' Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments without applying the
            softmax net activation function ??? it will not affect the most active neuron.
        '''
        y_net_act = (features @ self.y_wts) + self.y_b

        y_net_act[y_net_act<0] = 0
        
        z_net_in = y_net_act @ self.z_wts + self.z_b

        y_pred = np.argmax(z_net_in, axis=1)
        return y_pred

    def forward(self, features, y, reg=0):
        '''
        Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity
        to the output layer, ending with the cross-entropy loss computation.
        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all in one method
        is not good design, but as you will discover, having the
        forward computations (y_net_in, y_net_act, etc) easily accessible in one place makes the
        backward pass a lot easier to track during implementation. In future projects, we will
        rely on better OO design.

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        # mostly re-used from softmax.py, adjusted to add loss regularization for multiple layers

        # y layer, ReLU activation
        y_net_in = features @ self.y_wts + self.y_b
        y_net_act = np.maximum(0, y_net_in)

        # z layer, softmax activation
        z_net_in = y_net_act @ self.z_wts + self.z_b
        new_z_net_in = z_net_in + (-1*np.max(z_net_in, keepdims=True))
        z_net_act = np.exp(new_z_net_in) / (np.sum(np.exp(new_z_net_in), keepdims=True, axis=1))

        # loss
        correct = z_net_act[np.arange(z_net_act.shape[0]), y]


        loss = -np.mean(np.log(correct)) + (reg/2 * np.sum(self.y_wts ** 2)) + (reg/2 * np.sum(self.z_wts ** 2))

        return y_net_in, y_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''
        Performs a backward pass (output -> hidden -> input) during training to update the
        weights. This function implements the backpropogation algorithm.

        This should start with the loss and progate the activity
        backwards through the net to the input-hidden weights.

        I added dz_net_act for you to start with, which is your cross-entropy loss gradient.
        Next, tackle dz_net_in, dz_wts and so on.

        I suggest numbering your forward flow equations and process each for
        relevant gradients in reverse order until you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights!
        (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Regularize each layer's weights like usual.
        '''
        # 5-4 loss WRT netAct
        dz_net_act = -1/(len(z_net_act) * z_net_act)
        # 4-3, get one hot labels for z layer
        z_one_hot = self.one_hot(y, z_net_in.shape[1])
        dz_net_in = dz_net_act * z_net_act * (z_one_hot - z_net_act)
        # 3
        dz_wts = (dz_net_in.T @ y_net_act).T + (reg * self.z_wts)
        # 3
        dz_b = np.sum(dz_net_in, axis=0)
        # 3-2
        dy_net_act = dz_net_in @ self.z_wts.T
        # 2-1, help w/dy_net_in from Ethan S. and Nhi T.
        dy_net_in = dy_net_act * np.where(y_net_act <= 0, 0, 1)
        # 1
        dy_wts = (dy_net_in.T @ features).T + (reg * self.y_wts)
        # 1
        dy_b = np.sum(dy_net_in, axis=0)

        return dy_wts, dy_b, dz_wts, dz_b

    def fit(self, features, y, x_validation, y_validation,
            resume_training=False, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0, verbose=2,
            print_every=100):
        ''' Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each
            epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        resume_training: bool.
            False: we clear the network weights and do fresh training
            True: we continue training based on the previous state of the network.
                This is handy if runs of training get interupted and you'd like to continue later.
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the
            model per training epoch before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            on epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats.
            Recorded training loss on every epoch for the current mini-batch.
        train_acc_history: Python list of floats.
            Recorded accuracy on every training epoch on the current training mini-batch.
        validation_acc_history: Python list of floats.
            Recorded accuracy on every epoch on the validation set.

        TODO:
        -----------
        The flow of this method should follow the one that you wrote in softmax_layer.py.
        The main differences are:
        1) Remember to update weights and biases for all layers!
        2) At the end of an epoch (calculated from iterations), compute the training and
            validation set accuracy. This is only done once every epoch because "peeking" slows
            down the training.
        3) Add helpful printouts showing important stats like num_epochs, num_iter/epoch, num_iter,
        loss, training and validation accuracy, etc, but only if verbose > 0 and consider `print_every`
        to control the frequency of printouts.
        '''
        num_samps, num_features = features.shape
        #1 wts and bias
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        self.initialize_wts(self.num_input_units, self.num_hidden_units, self.num_output_units)
    
        

        #2 implementing mini batch 
        iterations = (int(np.round(num_samps/mini_batch_sz)))
        for i in range(n_epochs*iterations):
            #make an array that can index mini_batch_sz samples randomly from feats
         
            batch_indices = np.zeros((num_samps))
            batch_indices[:mini_batch_sz] = 1
            np.random.shuffle(batch_indices)
            batch = features[batch_indices.astype('bool')]
            batch_labels = y[batch_indices.astype('bool')]
            


            #forward pass:

            y_net_in, y_net_act, z_net_in, z_net_act, loss = self.forward(batch, batch_labels, reg) 
            loss_history.append(loss)


            #backward pass:

            dy_wts, dy_b, dz_wts, dz_b = self.backward(batch, batch_labels, y_net_in, y_net_act, z_net_in, z_net_act, reg)
            
            #8 gradient and bias
            self.y_wts  -= dy_wts*lr
            self.z_wts  -= dz_wts*lr
            self.y_b    -= dy_b*lr
            self.z_b    -= dz_b*lr

            if (i%iterations) == 0:
            #compute validation and training set accuracy.
                train_acc = self.accuracy(batch_labels, self.predict(batch))
                val_acc = self.accuracy(y_validation, self.predict(x_validation))
            
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
            
                if verbose >0:
                    if (i%print_every) == 0:
                        print(f"completed epoch on itr {i}/{n_epochs*iterations}, loss of {np.round(loss,3)}")
                        print(f"===> val_acc was {np.round(val_acc,4)}")

        return loss_history, train_acc_history, val_acc_history

