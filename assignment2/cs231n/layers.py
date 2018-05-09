import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension C.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, C)
    - b: A numpy array of biases, of shape (C,)

    Returns a tuple of:
    - out: output, of shape (N, C)
    - cache: (x, w, b)
    """
    cache = (x, w, b)

    X = x.reshape(x.shape[0], -1)
    out = np.dot(X, w) + b

    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, C)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, C)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, C)
    - db: Gradient with respect to b, of shape (C,)
    """
    x, w, b = cache
    X = x.reshape(x.shape[0], -1)
    dw = np.dot(X.T, dout) # (D, C)
    dx = np.dot(dout, w.T).reshape(*x.shape) # (N, d...)
    db = np.sum(dout, axis = 0) # (C,)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx = dout.copy()
    dx[cache <= 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        # 1) Calculate mean on each dimension across all samples. (D,)
        mu = np.mean(x, axis=0)

        # 2) Subtract mean vector of every sample
        xmu = x - mu

        # 3) calculate variance and add eps for numerical stability
        var = np.sum(xmu ** 2, axis = 0) / N
        vare = var + eps

        # 4) Sqrt and invert
        sqivare = vare ** -0.5

        # 5) normalization
        normx = xmu * sqivare

        # 6) Scale
        out = gamma * normx + beta

        # Update the running mean and variance with the momentum
        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        # Save values for back propogation
        cache = (gamma, xmu, vare, sqivare, normx)

    elif mode == 'test':
        normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    gamma, xmu, vare, sqivare, normx = cache

    N = dout.shape[0]

    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(dout * normx, axis = 0)

    dnormx = dout * gamma

    dxmu1 = dnormx * sqivare

    dsqivare = np.sum(dnormx * xmu, axis = 0)
    dsqrtdimvars = -0.5 * (vare**-1.5) * dsqivare
    dxmu2 = 2/N * xmu * dsqrtdimvars

    dx1 = dxmu1 + dxmu2
    dmu = -np.sum(dx1, axis = 0)

    dx2 = dmu / N
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    gamma, xmu, vare, sqivare, normx = cache

    N = dout.shape[0]

    dbeta = np.sum(dout, axis = 0)
    dgamma = np.sum(dout * normx, axis = 0)

    dnormx = dout * gamma

    dxmu1 = dnormx * sqivare
    dxmu2 = -1/N * xmu * (vare**-1.5) * np.sum(dnormx * xmu, axis = 0)

    dx1 = dxmu1 + dxmu2
    dmu = -np.sum(dx1, axis = 0)

    dx2 = dmu / N
    dx = dx1 + dx2

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = x
    if mode == 'train' and p > 0:
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    p, mode = dropout_param['p'], dropout_param['mode']

    dx = dout
    if mode == 'train' and p > 0:
        dx = dout * mask

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height fH and width fW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, fH, fW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, out_H, out_W) where out_H, out_W are given by
      out_H = 1 + (H + 2 * pad - fH) / stride
      out_W = 1 + (W + 2 * pad - fW) / stride
    - cache: (x, w, b, conv_param)
    """
    F, C, fH, fW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Pad the (w,h) tensor input
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)
    N, C, H, W = x_pad.shape

    out_H = int(1 + (H - fH) / stride)
    out_W = int(1 + (W - fW) / stride)
    out = np.zeros((N, F, out_H, out_W))

    weights_reshaped = w.reshape(F, -1).T
    for fh in range(out_H):
        start_h = fh * stride
        for fw in range(out_W):
            start_w = fw * stride
            # In this loop, filter-windowed x_pad has shape (N, C, fH, fW) and weights has (F, C*fH*fW). Output (N, F)
            out_reshaped = x_pad[:, :, start_h:start_h + fH, start_w:start_w + fW].reshape(N, -1).dot(weights_reshaped)
            out[:, :, fh, fw] = out_reshaped + b  # b (F, ) is brodcasted to all N samples

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives of shape (N, F, out_H, out_W)
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x. (N, C, H, W)
    - dw: Gradient with respect to w. (F, C, fH, fW)
    - db: Gradient with respect to b. (F, )
    """
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant', constant_values=0)

    N, C, H, W = x.shape
    N, F, out_H, out_W = dout.shape
    F, C, fH, fW = w.shape

    # Drop axis 0 and (out_H, out_W) to get shape (F, )
    db = np.sum(dout, axis=(0, 2, 3))

    dw = np.zeros((F, C, fH, fW))
    for ih in range(0, fH):
        for iw in range(0, fW):
            # In this loop, filter-windowed x_pad has shape (N, C, out_H, out_W) and dout has (N, F, out_H, out_W). We want (F, C) weight values.
            # So, reshape x_pad to (C, N*out_H*out_W) and dout to (F, N*out_H*out_W)
            dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
            x_pad_reshaped = x_pad[:, :, ih:(ih + out_H * stride):stride, iw:(iw + out_W * stride):stride].transpose(1, 0, 2, 3).reshape(C, -1)
            dw[:, :, ih, iw] = dout_reshaped.dot(x_pad_reshaped.T)

    # A dx value in a channel is the sum of the associated output value across all filters mulitplied by all the filters in the weights in a channel
    dx_pad = np.zeros(x_pad.shape)
    for oh in range(out_H):
        start_h = oh * stride
        for ow in range(out_W):
            start_w = ow * stride
            # In this loop, dout has shape (N, F) and w has reshape (F, C*fH*fW). Output (N, C*FH*fW)
            dx_pad[:, :, start_h:start_h + fH, start_w:start_w + fW] = dout[:, :, oh, ow].dot(w.reshape(F, -1)).reshape(N, C, fH, fW)

    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data of shape (N, C, out_H, out_W)
    - cache: (x, pool_param)
    """

    N, C, H, W = x.shape
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']
    out_H = int((H - pool_H) / stride + 1)
    out_W = int((W - pool_W) / stride + 1)

    out = np.zeros((N, C, out_H, out_W))
    mask = np.zeros((N, C, H, W), dtype=int)
    max_indices = np.zeros((out_H, out_W, N * C), dtype=int)
    for h in range(out_H):
        start_h = h * stride
        for w in range(out_W):
            start_w = w * stride
            window = x[:, :, start_h:start_h + pool_H, start_w:start_w + pool_W]
            max_indices[h, w] = np.argmax(window.reshape(N*C, pool_H*pool_W), axis=1)
            out[:, :, h, w] = np.max(window, axis=(2, 3)) # Keep (N, C). Max across all windows (h, w)

    # Save max_indices for use in backprop.
    pool_param['max_indices'] = max_indices
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives of shape (N, C, out_H, out_W)
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    N, C, out_H, out_W = dout.shape
    x, pool_param = cache
    pool_H = pool_param['pool_height']
    pool_W = pool_param['pool_width']
    stride = pool_param['stride']
    max_indices = pool_param['max_indices']
    N, C, H, W = x.shape

    dx = np.zeros((N, C, H, W))
    for h in range(out_H):
        start_h = h * stride

        for w in range(out_W):
            start_w = w * stride

            dx_reshaped = np.zeros((N*C, pool_H*pool_W))
            dx_reshaped[range(N*C), max_indices[h, w]] = dout[:, :, h, w].reshape(N*C)
            dx[:, :, start_h:start_h + pool_H, start_w:start_w + pool_W] = dx_reshaped.reshape(N, C, pool_H, pool_W)

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    x_reshaped = x.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose((0, 2, 3, 1)).reshape(N * H * W, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward_alt(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    # Taking exponentials may result in large numbers and diving by it could
    # be numerically unstable. Normalize by adding a log constant.
    # logC=−max x_j. So highest value in each sample = 0.
    normx = x - np.max(x, axis=1, keepdims=True)

    exps = np.exp(normx)
    deno = np.sum(exps, axis=1, keepdims=True)
    softmax = exps / deno

    loss = -np.sum(np.log(np.choose(y, softmax.T))) / N

    # Calculate derivative of loss with respect to the input x
    one_hot_encoded = np.zeros_like(x)
    one_hot_encoded[np.arange(N), y] = 1
    dx = (softmax - one_hot_encoded) / N
    return loss, dx
