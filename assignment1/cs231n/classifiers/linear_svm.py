import numpy as np


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  # Initialize loss and the gradient of W to zero
  dW = np.zeros(W.shape)
  loss = 0.0
  num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1

  # Compute the data loss and the gradient
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    num_classes_insufficient_score = 0

    for j in range(num_classes):
      if j == y[i]: # No loss if current class is X[i]'s target
        continue

      margin = scores[j] - correct_class_score + delta
      # Only calculate loss and gradient if margin condition is violated
      if margin > 0:
        loss += margin

        num_classes_insufficient_score += 1
        # Gradient update for incorrect class weight
        dW[:, j] += X[i]

    # Gradient update for correct class weight
    dW[:, y[i]] += (-X[i] * num_classes_insufficient_score)

  # Average our data loss across the batch
  loss /= num_train
  dW /= num_train

  # Add regularization loss to the data loss
  loss += reg * np.sum(W * W)
  # Add gradient of regularization term
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  # Handy re-usable values
  delta = 1
  num_train = X.shape[0]
  train_indices = np.arange(num_train)

  # Calculate all scores and select score at the actual target
  scores = np.dot(X, W)
  correct_scores = np.choose(y, scores.T)

  # SVM loss function wants the score of the correct class y to be larger than the
  # incorrect class scores by at least delta. If this is not the case, we will accumulate loss.
  # Calculate the SVM margin and keep bools of where scores did not satisfy the margin condition
  margins = scores - correct_scores[:, np.newaxis] + delta # (N, C)
  margins[train_indices, y] = 0 # Actual image target does not have a margin. Reset to 0
  has_insufficient_score = margins > 0 # (N, C)

  # Calculate loss, average across the batch and add regularization
  loss = np.sum(margins[has_insufficient_score])
  loss = loss / num_train + reg * np.sum(W * W)

  # Now, we calculate the gradients.
  # For each image, count the number of classes that had insufficient score
  num_classes_insufficient_score = np.sum(has_insufficient_score, axis=1) # (N, 1)

  # Classes that do not belong to the image will scale by 1 if it had an insufficient score
  gradients_scaling = np.zeros(margins.shape)
  gradients_scaling[has_insufficient_score] = 1
  # Image's actual class scales by -num_classes that had an insufficient score
  gradients_scaling[train_indices, y] = -num_classes_insufficient_score

  # Gradients dW is the data X scaled by the number of insufficient score classes
  dW = np.dot(X.T, gradients_scaling)
  # Average gradients across the batch and add gradient of regularization term
  dW = dW / num_train + 2 * reg * W

  return loss, dW
