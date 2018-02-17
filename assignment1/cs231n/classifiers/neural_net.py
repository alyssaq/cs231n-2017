from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.softmax import softmax_loss_vectorized


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def softmax(self, scores):
    # Pass scores through softmax and returns with same shape as scores
    return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  def activate(self, X):
    # ReLU
    return np.maximum(np.zeros(X.shape), X)

  def loss(self, X, y=None, reg=0.0, idropout_prob=1.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.
    - idropout_prob: Inverted dropout probability. Higher = less dropout

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass.
    L1 = self.activate(np.dot(X, W1) + b1) # (N, H)
    if idropout_prob > 0 and idropout_prob < 1:
       # Dropout mask. /p for scaling
      L1 *= ((np.random.rand(*L1.shape) < idropout_prob) / idropout_prob)
    scores = np.dot(L1, W2) + b2 # (N, C)

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # To make the softmax function numerically stable, normalize the values
    # by multiplying the numerator and denominator with a constant C
    scores -= np.max(scores, axis=1)[:, np.newaxis]

    # Compute the softmax loss with the scores (N, C)
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    # Calculate the cross entropy loss summed across the sample
    loss = np.sum(-np.log(np.choose(y, softmax.T)))
    # Average loss across batch size and add regularization for both weights
    loss = loss / N + (reg * np.sum(W1 * W1)) + (reg * np.sum(W2 * W2))

    #############################################################################
    # Compute the backward pass, computing the derivatives of the weights       #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    # Use the Softmax classifier loss.
    grads = {}

    # One hot encode labels (1 where the index is the true target)
    one_hot_labels = np.zeros(softmax.shape)
    one_hot_labels[np.arange(N), y] = 1

    # http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    # Calculate gradient dL/dW = X.T * dL/dY (cross-entropy loss output wrt its input)
    dscores = (softmax - one_hot_labels) / N # (N, C)
    grads['W2'] = np.dot(L1.T, dscores)
    grads['b2'] = np.sum(dscores, axis = 0)
    grads['W2'] += reg * W2 * 2

    # Derivative of L1 based on derivate of softmax scores
    dL1 = np.dot(dscores, W2.T)
    dL1[L1 <= 0] = 0
    grads['W1'] = np.dot(X.T, dL1)
    grads['b1'] = np.sum(dL1, axis = 0)
    grads['W1'] += reg * W1 * 2

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, idropout_prob=1.0, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - idropout_prob: Inverted dropout probability. Higher = less dropout
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    train_indices = np.arange(num_train)
    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      # Sample batch_size elements from the training data and their
      # corresponding labels to use in this round of gradient descent.
      batch_indices = np.random.choice(train_indices, batch_size)
      X_batch = X[batch_indices]
      y_batch = y[batch_indices]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg, idropout_prob=idropout_prob)
      loss_history.append(loss)

      # Update parameters of the network - weights and biases
      for parameter in self.params:
        self.params[parameter] -= grads[parameter] * learning_rate

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    # Compute one forward pass with our tuned parameters
    L1 = self.activate(np.dot(X, W1) + b1) # (N, H)
    scores = np.dot(L1, W2) + b2 # (N, C)

    # The predicted class is where we have the highest score
    y_pred = np.argmax(scores, axis=1)

    return y_pred

  def accuracy(self, y, y_pred):
    return np.mean(y == y_pred)
