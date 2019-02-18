"""
Damanpreet Kaur
"""


#from __future__ import division
#from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from utils.util_functions import LinearTransform, ReLU, SigmoidCrossEntropy, create_random_mini_batches, MLP


if __name__ == '__main__':
    time_in = time.time()
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = np.array(data[b'train_data'], dtype = 'float64') 
    train_y = data[b'train_labels']
    test_x = data[b'test_data'] 
    test_y = data[b'test_labels']

    # normalize the datasets 
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    num_examples, input_dims = train_x.shape
    
	# Hyper-parameters
    num_epochs = 100
    hidden_units = 100
    learning_rate = 0.01
    momentum = 0.8
    l2_penalty = 0.001
    keep_probability = 0.8
    total_error_cnt_train = []
    total_loss_ep_train = []
    total_error_cnt_test = []
    total_loss_ep_test = []

    mlp = MLP(input_dims, hidden_units)

    for epoch in range(num_epochs):
        if epoch%10 == 0 and epoch <= 20:
            learning_rate = learning_rate * math.pow(0.5, math.floor((1+epoch)/10.0))
            print('Updated learning rate is ', learning_rate)
        if epoch%5 == 0 and epoch > 30:
            learning_rate = learning_rate * math.pow(0.8, math.floor((1+epoch)/10.0))

        # random shuffle and create mini-batches
        total_error = 0
        total_loss = 0.0
        total_error_test = 0
        total_loss_test = 0.0
        mini_batches, num_batches = create_random_mini_batches(train_x, train_y, None)

        for b in range(num_batches):

            mini_batch_X = mini_batches[b][0]
            mini_batch_Y = mini_batches[b][1]

            W1, W2, b1, b2, A2, loss, error_count = mlp.train(mini_batch_X, mini_batch_Y, learning_rate, momentum, l2_penalty, keep_probability)
            
            total_error += error_count
            total_loss += loss

            # MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
        
        total_loss_test, total_error_test = mlp.evaluate(test_x, test_y)

        total_error_cnt_train.append(total_error)
        total_loss_ep_train.append(total_loss/num_batches)
        total_error_cnt_test.append(total_error_test)
        total_loss_ep_test.append(total_loss_test)
        
        print('Loss for epoch ', epoch+1,': ', total_loss/num_batches)
        print('Error count for epoch', epoch+1, ': ', total_error)
        print('Loss for epoch on test data: ', total_loss_test)
        print('Error count for epoch: ',total_error_test)
    
    mlp.model_forward_prop(train_x)
    train_loss = mlp.calculate_loss(train_y, l2_penalty)
    prediction = mlp.prediction()
    result = (prediction == train_y)
    error_count = len((np.where(result==False))[0])
    train_accuracy = 100. * (num_examples - error_count)/num_examples    

    test_total_exs = test_x.shape[0]
    mlp.model_forward_prop(test_x)
    test_loss = mlp.calculate_loss(test_y, l2_penalty)
    prediction = mlp.prediction()
    result = (prediction == test_y)
    error_count = len((np.where(result==False))[0])
    test_accuracy = 100. * (test_total_exs - error_count)/test_total_exs

    print(train_loss)
    print(test_accuracy)

    # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
    print()
    print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
        np.squeeze(train_loss),
        np.squeeze(train_accuracy) ,
    ))
    print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        np.squeeze(test_loss),
        np.squeeze(test_accuracy),
    ))

    train_accuracy = [100 * (num_examples - error_cnt)/ num_examples for error_cnt in total_error_cnt_train]
    test_accuracy = [100 * (test_total_exs - error_cnt)/ test_total_exs for error_cnt in total_error_cnt_test]
    plt.figure()
    plt.xticks(np.arange(0, len(train_accuracy), 5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of training and testing data')
    plt.plot(train_accuracy, label = 'Train Accuracy')
    plt.plot(test_accuracy, label = 'Test Accuracy')
    plt.legend()
    plt.savefig('Accuracy.png')

    plt.figure()
    plt.xticks(np.arange(0, len(total_error_cnt_train), 5))
    plt.yticks(np.arange(0, max(total_error_cnt_train), 200))
    plt.xlabel('Epoch')
    plt.ylabel('Error count')
    plt.title('Misclassification error rate')
    plt.plot(total_error_cnt_train, label = 'Train error rate')
    plt.plot(total_error_cnt_test, label = 'Test error rate')
    plt.legend()
    plt.savefig('Error_count_batch.png')

    plt.figure()
    plt.xticks(np.arange(0, len(total_loss_ep_train), 5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.plot(total_loss_ep_train, label = 'Train loss')
    plt.plot(total_loss_ep_test, label = 'Testing loss')
    plt.legend()
    plt.savefig('Loss_batch.png')
    
    time_out = time.time()
    print('Total time taken for execution: ', time_out - time_in)
