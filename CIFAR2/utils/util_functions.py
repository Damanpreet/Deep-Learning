import numpy as np
from scipy.special import expit
import math


def create_random_mini_batches(X, Y, num_batches = None, mini_batch_size = 64):
    '''
        This function creates random mini-batches.
    '''
    num_examples = X.shape[0]
    mini_batches = []
    permutations = list(np.random.permutation(num_examples))
    X_shuffled = X[permutations, :]
    Y_shuffled = Y[permutations, :]

    if num_batches == None:
        num_batches = math.floor(num_examples/mini_batch_size)

        for i in range(num_batches):
            mini_batch_X = X_shuffled[mini_batch_size*i:mini_batch_size*(i+1),:]
            mini_batch_Y = Y_shuffled[mini_batch_size*i:mini_batch_size*(i+1),:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        if num_examples%mini_batch_size != 0:
            mini_batch_X = X_shuffled[mini_batch_size*num_batches:,:]
            mini_batch_Y = Y_shuffled[mini_batch_size*num_batches:,:]
            mini_batches.append((mini_batch_X, mini_batch_Y))
            num_batches += 1

    else:
        mini_batch_size = math.ceil(num_examples/num_batches)   

        for i in range(num_batches-1):
            mini_batch_X = X_shuffled[mini_batch_size*i:mini_batch_size*(i+1),:]
            mini_batch_Y = Y_shuffled[mini_batch_size*i:mini_batch_size*(i+1),:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        mini_batch_X = X_shuffled[mini_batch_size*(num_batches-1):,:]
        mini_batch_Y = Y_shuffled[mini_batch_size*(num_batches-1):,:]
        mini_batches.append((mini_batch_X, mini_batch_Y))   

    return mini_batches, num_batches


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.A_prev = x
        self.Z = np.dot(self.A_prev, self.W) + self.b
        return self.Z

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        dW = np.dot(self.A_prev.T, grad_output) + l2_penalty * self.W
        db = np.sum(grad_output, axis = 0, keepdims=True) 
        dA = np.dot(grad_output, self.W.T)
        return dA, dW, db 


# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        self.A1 = np.maximum(0,x)
        return self.A1 

    # DEFINE backward function
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        dZ1 = np.zeros((grad_output.shape))
        dZ1[self.A1 > 0] = 1
        dZ1[self.A1 == 0] = np.random.uniform(0.01,1)
        dZ1[self.A1 < 0] = 0
        dZ1 = np.multiply(dZ1, grad_output)
        return dZ1


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
    # DEFINE forward function
        self.A2 = expit(x)
        return self.A2


    # DEFINE backward function
    def backward(
	    self, 
	    y_batch, 
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):
        dZ2 = self.A2 - y_batch
        return dZ2


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
        self.W1 = np.random.randn(input_dims, hidden_units) * 0.01 
        self.b1 = np.zeros((1, hidden_units))
        
        self.W2 = np.random.randn(hidden_units, 1) * 0.01  
        self.b2 = np.zeros((1, 1))

        self.VdW1 = np.zeros((self.W1.shape))
        self.VdW2 = np.zeros((self.W2.shape))
        self.Vdb1 = np.zeros((self.b1.shape))
        self.Vdb2 = np.zeros((self.b2.shape))


    def model_forward_prop(self, x_batch, keep_probs = 0.8):
        # Hidden unit
        self.layer1_lin = LinearTransform(self.W1, self.b1)        
        self.Z1 = self.layer1_lin.forward(x_batch)
        self.layer1_relu = ReLU()
        self.A1 = self.layer1_relu.forward(self.Z1)

        # applying dropout
        self.D1 = np.random.rand(self.A1.shape[0], self.A1.shape[1])
        self.D1 = self.D1 < keep_probs
        self.A1 = np.multiply(self.D1, self.A1)
        self.A1 = self.A1 / keep_probs

        # Output unit
        self.layer2_lin = LinearTransform(self.W2, self.b2)
        self.Z2 = self.layer2_lin.forward(self.A1)
        self.layer2_sigm = SigmoidCrossEntropy()
        self.A2 = self.layer2_sigm.forward(self.Z2)


    def model_backward_prop(self, y_batch, l2_penalty, keep_probs = 0.8):
        m = y_batch.shape[0]
        dZ2 = self.layer2_sigm.backward(y_batch) 
        dA1, self.dW2, self.db2 = self.layer2_lin.backward(dZ2, l2_penalty)
        self.dW2, self.db2 = (1.0/m) * self.dW2 , (1.0/m) * self.db2
        
        # applying dropout
        dA1 = np.multiply(self.D1, dA1)
        dA1 = dA1 / keep_probs

        dZ1 = self.layer1_relu.backward(dA1)
        dA0, self.dW1, self.db1 = self.layer1_lin.backward(dZ1, l2_penalty)
        self.dW1, self.db1 = (1.0/m) * self.dW1, (1.0/m) * self.db1


    # Predict result
    def prediction(self):
        prediction = np.zeros((self.A2.shape))
        prediction[self.A2 > 0.5] = 1
        prediction[self.A2 <= 0.5] = 0
        return prediction

    # Calculate error
    def calculate_error(self, y_batch):
        prediction = self.prediction()
        result = (prediction == y_batch)
        self.error_count = len((np.where(result==False))[0])

    # Update parameters with learning rate and momentum
    def update_parameters(self, learning_rate, momentum):
        self.VdW1 = momentum * self.VdW1 - learning_rate * self.dW1
        self.VdW2 = momentum * self.VdW2 - learning_rate * self.dW2
        self.Vdb1 = momentum * self.Vdb1 - learning_rate * self.db1
        self.Vdb2 = momentum * self.Vdb2 - learning_rate * self.db2
        
        self.W1 = self.W1 + self.VdW1
        self.W2 = self.W2 + self.VdW2
        self.b1 = self.b1 + self.Vdb1
        self.b2 = self.b2 + self.Vdb2


    # L2 regularization cost
    def calculate_loss(self, y_batch, l2_penalty):
        L2_reg_cost = (l2_penalty / (2 * y_batch.shape[0])) * (np.sum((self.W1)**2) + np.sum((self.W2)**2))
        pos_log = np.log(self.A2 + 1e-15)
        neg_log = np.log(1 - self.A2 + 1e-15)
        self.loss =  (np.multiply(y_batch, pos_log) + np.multiply((1 - y_batch), neg_log)) 
        self.loss = -1*np.mean(self.loss, axis = 0) 
        self.loss += L2_reg_cost
        return self.loss


    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
        keep_probability = 0.8
    ):
	# INSERT CODE for training the network
        ''' 
            1. Forward propogation 
            2. Calculate the loss
            3. Model Backward prop to compute the gradients
            4. Update the parameters
            5. Check the error count for each mini-batch.
        '''
        self.model_forward_prop(x_batch, keep_probability)
        loss = self.calculate_loss(y_batch, l2_penalty)
        self.model_backward_prop(y_batch, l2_penalty, keep_probability)
        self.update_parameters(learning_rate, momentum)
        self.calculate_error(y_batch)
  
        return self.W1, self.W2, self.b1, self.b2, self.A2, np.squeeze(self.loss), self.error_count

    def evaluate(self, x, y):
	# INSERT CODE for testing the network
        '''
            This function returns the error count and loss on test data.
            1. Run forward prop on test data with the final weights of each epoch.
            2. Compute the loss
            3. Calculate the error 
        '''
        self.model_forward_prop(x)

        pos_log = np.log(self.A2 + 1e-15)
        neg_log = np.log(1 - self.A2 + 1e-15)
        loss =  (np.multiply(y, pos_log) + np.multiply((1 - y), neg_log)) 
        loss = -1*np.mean(loss, axis = 0) 

        self.calculate_error(y)

        return np.squeeze(loss), self.error_count
