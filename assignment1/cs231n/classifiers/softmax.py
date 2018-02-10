import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax cross-entropy loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W) # (1, C)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]

    # Cross entropy loss
    softmax = np.exp(scores) / np.sum(np.exp(scores), keepdims=True)
    loss += -np.log(softmax[y[i]])

    for j in range(num_classes):
      if j == y[i]:
        dW[:, j] += (softmax[j] - 1) * X[i].T
      else:
        dW[:, j] += (softmax[j]) * X[i].T

  # Average our data loss across the batch
  loss /= num_train
  dW /= num_train

  # Add regularization loss to the data loss
  loss += reg * np.sum(W * W)
  # Add gradient of regularization term
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  train_indices = np.arange(num_train)

  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:, np.newaxis]

  # Pass scores through softmax (N, C)
  softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  # Calculate the cross entropy loss summed across the sample and add regularization
  loss = np.sum(-np.log(np.choose(y, softmax.T)))
  loss = loss / num_train + reg * np.sum(W * W)

  # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
  # The derivate dL/dW = dL/dS * dS/dW where S = softmax
  # One hot encode labels (Kronecker delta - 1 where index = target, 0 otherwise)
  one_hot_labels = np.zeros(softmax.shape)
  one_hot_labels[train_indices, y] = 1

  # Calculate gradient and add derivative of regularization
  gradient_multipler = softmax - one_hot_labels
  dW = np.dot(X.T, gradient_multipler)
  dW = dW / num_train + 2 * reg * W

  return loss, dW

