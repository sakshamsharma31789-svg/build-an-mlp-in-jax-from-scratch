"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    # TODO: wrap a Python integer seed into a JAX PRNG key (uint32 array of shape (2,))
    return jax.random.PRNGKey(seed)
    pass

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    # TODO: split `key` into `num` independent subkeys and return them as a (num, 2) array.
    return jax.random.split(key,num)
    pass

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    return jax.random.normal(key,shape)
    pass

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return jax.random.normal(key,(batch_size,num_features))
    pass

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    # TODO: return an int32 label per row using the first num_classes feature columns.
    q = jnp.argmax(inputs[:,:num_classes],axis=-1)
    return jnp.int32(q)
    pass

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # TODO: Convert a 1-D array of integer class indices into a 2-D one-hot matrix of shape (batch, num_classes).
    return jnp.array(labels[:,None]== jnp.arange(num_classes),dtype=jnp.float32)
    pass

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    # TODO: sample W from a scaled normal and set b to zeros, return as a dict.
    bias = jnp.zeros((out_dim,))
    weight = sample_normal_matrix(key,(in_dim,out_dim))*scale
    return {'W':weight,'b':bias}
    pass

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # TODO: build a list of per-layer parameter dicts from adjacent layer sizes.
    num_layers = len(layer_sizes) - 1
    
    keys = jax.random.split(key, num_layers)
    
    params = []
    for k, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        layer_params = init_linear_layer(k, in_dim, out_dim, scale=scale)
        params.append(layer_params)
        
    return params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # TODO: compute x @ W + b using layer_params['W'] and layer_params['b'].
    return jnp.matmul(x,layer_params['W']) + layer_params['b']
    pass

# Step 10 - relu_activation
import jax.numpy as jnp


def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    # TODO: return an array of the same shape with negatives replaced by zero.
    return jnp.maximum(0,x)
    pass

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    # TODO: convert logits into a numerically stable softmax along the last axis
    max_logits = jnp.max(logits,axis=-1,keepdims=True)
    exp_logits = jnp.exp(logits-max_logits)
    return exp_logits/jnp.sum(exp_logits,axis= -1, keepdims = True)
    pass

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # TODO: run x through all hidden layers with ReLU, then a final linear layer, returning logits.
    for layers in params[:-1]:
        x = linear_forward(x,layers)
        x = relu_activation(x)
    out = linear_forward(x,params[-1])
    return out
    pass

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # TODO: return the numerically stable log-softmax of logits along the last axis.
    max_logits = jnp.max(logits,axis=-1,keepdims =True)
    stabilized_logits = logits - max_logits
    sum_exp = jnp.sum(jnp.exp(stabilized_logits),axis=-1,keepdims=True)
    return stabilized_logits - jnp.log(sum_exp)
    pass

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    # TODO: return the mean cross-entropy between logits and one-hot targets
    log_probs = log_softmax_logits(logits)
    return -jnp.mean(jnp.sum(one_hot_targets*log_probs,axis=-1))
    pass

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    # TODO: compute predicted classes from logits and compare to labels
    preds = jnp.argmax(logits,axis=-1)
    return jnp.mean(preds == labels)

    pass

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    # TODO: return scalar cross-entropy loss as a function of params, ready for jax.grad
    logits = mlp_forward(params,x)
    loss = cross_entropy_loss(logits,one_hot_targets)
    return loss
    pass

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    # TODO: return grad of loss_fn_of_params w.r.t. params using jax.grad
    grad_fn = jax.grad(loss_fn_of_params)
    return grad_fn(params,x,one_hot_targets)
    pass

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    # TODO: apply one SGD step to every parameter using its gradient and a learning rate
    updated = []
    for layer_param,layer_grad in zip(params,grads):
        updated_layer = {
            'W':layer_param['W'] - learning_rate * layer_grad['W'],
            'b':layer_param['b'] - learning_rate * layer_grad['b']
        }
        updated.append(updated_layer)
    return updated
    pass

# Step 19 - training_step
import jax
import jax.numpy as jnp

def training_step(params, x, one_hot_targets, learning_rate):
    # TODO: compute current loss + grads via the upstream helpers, then SGD-update params.
    loss = loss_fn_of_params(params,x,one_hot_targets)
    grad = compute_param_grads(params,x,one_hot_targets)
    new = sgd_update_params(params,grad,learning_rate)
    return new,loss
    pass

# Step 20 - train_mlp
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    """Run num_epochs full-batch SGD updates and return the final params."""
    # TODO: run num_epochs full-batch SGD updates via training_step and return final params
    for _ in range(num_epochs):
        params,_ = training_step(params,x,one_hot_targets,learning_rate)
    return params
    pass

# Step 21 - predict_classes
def predict_classes(params, x):
    # TODO: run mlp_forward on x and return the argmax class index per row
    forward = mlp_forward(params,x)
    return jnp.argmax(forward,axis = -1)
    pass

