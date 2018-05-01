from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 conv_param=None, pool_param=None, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # W1 (F, C, HH, WW), b1 (F) for conv layer
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)

        if conv_param is None:
            conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
            self.conv_param = conv_param
        if pool_param is None:
            pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
            self.pool_param = pool_param

        pad = conv_param['pad']
        conv_stride = conv_param['stride']
        pool_size = pool_param['pool_height']
        pool_stride = pool_param['stride']

        conv_out_size = int(1 + (H + 2 * pad - filter_size) / conv_stride)
        pool_out_size = int((conv_out_size - pool_size) / pool_stride + 1)
        # Flatten the pooling output to prep as input into affine layer (D, C)
        pool_dim = num_filters * pool_out_size * pool_out_size

        # W2 (P, hidden_dim) where P=(F, pool_out_size, pool_out_size)
        self.params['W2'] = np.random.normal(0, weight_scale, (pool_dim, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        conv_param, pool_param = self.conv_param, self.pool_param
        N = X.shape[0]

        conv_out, cache_conv = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N, F, HH, WW = conv_out.shape
        conv_out_flatten = conv_out.reshape(N, -1) # Flatten for affine layer
        affine_hidden, cache_hidden = affine_relu_forward(conv_out_flatten, W2, b2)
        scores, cache_scores = affine_forward(affine_hidden, W3, b3)

        if y is None:
            return scores

        ############################################################################
        # Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reg = self.reg
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * ((reg * np.sum(W1 * W1)) + (reg * np.sum(W2 * W2)) + (reg * np.sum(W3 * W3)))

        grads = {}
        dH2, dW3, db3 = affine_backward(dscores, cache_scores)
        grads['W3'] = dW3 + (reg * W3)
        grads['b3'] = db3

        dH1, dW2, db2 = affine_relu_backward(dH2, cache_hidden)
        grads['W2'] = dW2 + (reg * W2)
        grads['b2'] = db2

        # Reshape the output from flatten affine layer into a conv-filter shape
        dx, dW1, db1 = conv_relu_pool_backward(dH1.reshape(N, F, HH, WW), cache_conv)
        grads['W1'] = dW1 + (reg * W1)
        grads['b1'] = db1

        return loss, grads
