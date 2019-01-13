import sys
sys.path.append(r"C:\Users\bradley.tjandra\Dropbox\Current Roles\Machine Learning\EA AI Meetups\DeepCompression")

import helper
import tensorflow as tf
import numpy as np
from scipy.special import logsumexp
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import math
import datetime
#import os
#from collections import Counter
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

model_dir = r"C:\Users\bradley.tjandra\Documents\AI\SoftWeightSharing"    
#pd.options.display.float_format = '{:.4e}'.format

# general data
X_train=train_data
Y_train=train_labels
X_test=test_data
Y_test=test_labels

#X_batch, Y_batch = helper.get_minibatch([X_train,Y_train],constants["minibatch_size"],1)

def create_placeholders(params):

    n_H = params["n_H"]
    n_W = params["n_W"]
        
    placeholders = { 
            'features' : tf.placeholder(tf.float32, shape=(None, n_H, n_W), name="place_X"),
            'labels' : tf.placeholder(tf.int64, shape=(None), name="place_Y"),
            'comp_switch' : tf.placeholder(tf.bool, name="comp_switch"),
            'tau' : tf.placeholder(tf.float32, name="tau"),
            'lambda_pi' : tf.placeholder(tf.float32, name="lambda_class_pi")
            }
    return placeholders
    

temp = tf.layers.Dense(units=300, activation=tf.nn.relu, name="tempname")
sess = tf.Session()
temp.get_weights()


def get_weights(layers):
        
#    weights = [L.trainable_weights for L in layers]
    
    weights = []
    flat_weights = []
#    units = [300,100]
    for l in range(len(layers)):
        lay_name = "lay"+str(l)+"/"
        w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=lay_name+"kernel:0")[0]
        b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=lay_name+"bias:0")[0]
        weights.extend([w, b])
##        with tf.variable_scope("lay"+str(l), reuse=True):
##            w = tf.get_variable("kernel")
##            b = tf.get_variable("bias")
        flat_weights.append(tf.reshape(w, [-1]))
        flat_weights.append(tf.reshape(b, [-1]))
            
    flat_weights = tf.concat(flat_weights,axis=0)
    
    return (flat_weights, weights)

def create_nn_model(params):
    
    units = params["units"]
    n_y = params["n_y"]
    
    layers = []
    for l in range(len(units)):
        layer = tf.layers.Dense(units=units[l], activation=tf.nn.relu, name="lay"+str(l))
        layers.append(layer)
        
    layers.append(tf.layers.Dense(n_y, activation=None, name="lay"+str(len(units))))
    return layers

def nn_model(placeholders, layers, params):
    
    features = placeholders['features']    
    units = params["units"]
    n_y = params["n_y"]
    n_H = params["n_H"]
    n_W = params["n_W"]
    
    net = tf.reshape(features, [-1, n_H * n_W])
    for L in layers:
        net = L.apply(net)
        
    return net

def create_compression_priors(params):
    
    n_classes = params["n_classes"]
    
    priors = {
            "pi"    : tf.Variable(tf.zeros(n_classes+1), name="pi"),
            "mu"    : tf.Variable(tf.zeros(n_classes+1), name="mu"),
            "gamma" : tf.Variable(tf.zeros(n_classes+1), name="gamma")}
    
    # create a tensor for the class prob, so that we can track this
    priors["class_prob"] = tf.exp(priors["pi"])
    
    return priors

def set_compression_priors(weights, priors, params):
    
    n_classes = params["n_classes"]
    p0 = params["p0"]
        
    mu = tf.linspace(tf.reduce_min(weights), tf.reduce_max(weights), n_classes)
    mu = tf.concat([tf.Variable([0.]), mu], axis=0) # as specified in the paper, mu0=0

    single_gamma = tf.log((tf.reduce_max(weights)-tf.reduce_min(weights))/float(n_classes)*1.0)
    single_gamma = tf.reshape(single_gamma,[1, 1])
    gamma = tf.tile(single_gamma, multiples = [1, n_classes+1])
    gamma = tf.reshape(gamma, [-1])
    
    single_pi = tf.log(1.-p0) - tf.log(float(n_classes))
    single_pi = tf.reshape(single_pi, [1,1])
    pi = tf.tile(single_pi, multiples = [1, n_classes])
    pi = tf.reshape(pi, [-1])
    pi = tf.concat([tf.Variable([tf.log(p0)]), pi], axis=0) # as specified in the paper
    
    priors_assign = {
            "pi"    : tf.assign(priors["pi"], pi),
            "mu"    : tf.assign(priors["mu"], mu),
            "gamma" : tf.assign(priors["gamma"], gamma),
            }
    
    return priors_assign

def calculate_compression(placeholders, weights, priors, params):
    
    # if called before prior parameters are set
    if priors == None:
        return 0.
    
    lam_pi = placeholders["lambda_pi"]
    
    n_classes = params["n_classes"]
    p0 = params["p0"]
    
    pi      = priors["pi"]
    gamma   = priors["gamma"]
    mu      = priors["mu"]
    prob    = priors["class_prob"]
    
    sigma2 = tf.exp(gamma)
    mu = tf.slice(mu,[1], [n_classes]) # drop the 0th class
    mu = tf.concat([tf.Variable([0.]), mu], axis=0) # as specified in the paper, mu0=0
    pi = tf.slice(pi,[1], [n_classes]) # drop the 0th class
    probs_excl_zeroth = tf.reduce_sum(tf.exp(pi))
    pi = tf.concat([tf.Variable([tf.log(p0)]), pi], axis=0) # as specified in the paper
    
    D2 = - 0.5 / sigma2 * tf.squared_difference(tf.expand_dims(weights,axis=-1), mu) 
    logL = pi - 0.5 * gamma + D2
    NLL = - tf.reduce_logsumexp(logL, axis=1)
    loss = tf.reduce_mean(NLL)
    loss = loss + lam_pi * tf.square(probs_excl_zeroth + p0 - 1) # Lagrangian to ensure probs sum to 1
    
    return (loss, NLL, D2)
    
def eval_model(placeholders, logits, weights, params, priors = None):
    
    labels = placeholders['labels']
    comp_switch = placeholders['comp_switch']
    tau = placeholders['tau']
    
    err_loss    = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    comp_loss, NLL, D2   = tf.cond(comp_switch, 
                          lambda : calculate_compression(placeholders, weights, priors, params),
                          lambda : (0., 0., 0.))
    loss = err_loss + tau * comp_loss
    
    preds = tf.argmax(logits, 1)
    accuracy = tf.metrics.accuracy(labels, preds)
    
    return (loss, err_loss, comp_loss, accuracy, NLL, D2)

def clip_pi(priors):
    
    pi = priors["pi"]
    clipped_pi = tf.minimum(pi, tf.zeros_like(pi))
    return tf.assign(pi, clipped_pi)
    

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def all_summaries(weights, priors):
    
    with tf.name_scope("weights"):
        variable_summaries(weights)
    with tf.name_scope("class_prob"):
        variable_summaries(priors["class_prob"])
    with tf.name_scope("mu"):
        variable_summaries(priors["mu"])
    with tf.name_scope("gamma"):
        variable_summaries(priors["gamma"])
        
    return tf.summary.merge_all()

def get_optimisation_op(trainable_variables ,learning_rates, loss):
    
    assert(len(trainable_variables) == len(learning_rates))
    optimizer = tf.train.AdamOptimizer(learning_rate=1.)
    grads_and_vars = optimizer.compute_gradients(loss, var_list = trainable_variables)
#    new_grads_and_vars = []
#    for i in range(len(grads_and_vars)):
#        lr = learning_rates[i]
#        grad = grads_and_vars[i][0]
#        var = grads_and_vars[i][1]
#        new_grads_and_vars.append((lr * grad, var))
#        
#    print(new_grads_and_vars)
#        
    grads_and_vars = [(r * gv[0], gv[1]) for r, gv in zip(learning_rates, grads_and_vars)]
    optimizer_op = optimizer.apply_gradients(grads_and_vars)
    return (optimizer, optimizer_op)
    
#    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rates[0])
#    return (optimizer, optimizer.minimize(loss, var_list = trainable_variables))
    

tf.reset_default_graph()
params = {
        "minibatch_size" : 300,
        "n_H" : 28,
        "n_W" : 28,
        "n_y" : 10,
        "units" : [300, 100],
        "n_classes" : 16,
        "p0" : 0.99
        }
placeholders = create_placeholders(params)
priors = create_compression_priors(params)
layers = create_nn_model(params)
logits = nn_model(placeholders, layers, params)
flat_weights, weights = get_weights(layers)
priors_assign = set_compression_priors(flat_weights, priors, params)
eval_tensors = eval_model(placeholders, logits, flat_weights, params, priors)
total_loss  = eval_tensors[0]
err_loss    = eval_tensors[1]
comp_loss   = eval_tensors[2]
accuracy    = eval_tensors[3]
summary = all_summaries(flat_weights, priors)
trainable_variables = [w for w in weights]
#trainable_variables.extend([priors["pi"], priors["mu"], priors["gamma"]])
#len(trainable_variables)
learning_rates = [0.001] * 6 + [5E-4] * 3
learning_rates = [0.001] * 6 
optimizer, optimizer_op = get_optimisation_op(
        trainable_variables,
        learning_rates,
        total_loss)
#optimizer = tf.train.AdamOptimizer(learning_rate=1)
#optimizer_op = optimizer.minimize(total_loss)
clip_pi_op = clip_pi(priors)

def set_minibatch(feed_dict, placeholders, data, params = None, i = None):
    
    if params == None:
        X, Y = data    
    else:
        minibatch_size = params["minibatch_size"]
        X, Y = helper.get_minibatch(data,minibatch_size,i)
        
    feed_dict[placeholders["features"]] = X
    feed_dict[placeholders["labels"]]   = Y
    return feed_dict

feed_dict = {
        placeholders["comp_switch"] : False,
        placeholders["tau"]         : 0.05,
        placeholders["lambda_pi"]   : 1
        }

weights_save = {}
sess = tf.Session()

#train_writer = tf.summary.FileWriter(r"C:\Users\bradley.tjandra\Documents\AI\SoftWeightSharing\Training\20190106_6", sess.graph)    
sess.run(tf.global_variables_initializer(),feed_dict={placeholders["comp_switch"] : True})
sess.run(tf.local_variables_initializer())
minibatch_size = params['minibatch_size']   

# training without compression
for epoch in range(30):
    
    data = helper.shuffle_in_unison([X_train, Y_train], index=0, seed=epoch)
    
    # minibatches
    for i in range(int(X_train.shape[0] / minibatch_size)):
        feed_dict = set_minibatch(feed_dict, placeholders, data, params, i)
        sess.run(optimizer_op, feed_dict = feed_dict)
    
    # test results
    feed_dict = set_minibatch(feed_dict, placeholders, [test_data, test_labels])
    summ_vals, eval_vals = sess.run((summary, eval_tensors), feed_dict = feed_dict)
    print("Test accuracy after epoch {} / 30: {}".format(epoch, eval_vals[3][1]))
    print("Loss: {}, Error: {}, Compression: {}".format(eval_vals[0], eval_vals[1], eval_vals[2]))
#    if epoch % 10 == 0:
#        train_writer.add_summary(summ_vals, epoch)

weights_save["Pre Compression"] = sess.run(flat_weights)

# training with compression
feed_dict[placeholders["comp_switch"]] = True
sess.run(priors_assign) 
for epoch in range(30):
    
    data = helper.shuffle_in_unison([X_train, Y_train], index=0, seed=epoch)
    
    # minibatches
    for i in range(int(X_train.shape[0] / minibatch_size)):
        feed_dict = set_minibatch(feed_dict, placeholders, data, params, i)
        _, total_loss_val = sess.run((optimizer_op, total_loss), feed_dict = feed_dict)
    
    # test results
    feed_dict = set_minibatch(feed_dict, placeholders, [test_data, test_labels])
    summ_vals, eval_vals = sess.run((summary,eval_tensors), feed_dict = feed_dict)
    sess.run(clip_pi_op)
    print("Test accuracy after epoch {} / 30: {}".format(epoch, eval_vals[3][1]))
    print("Loss: {}, Error: {}, Compression: {}".format(eval_vals[0], eval_vals[1], eval_vals[2]))
    if epoch % 5 == 0:
        train_writer.add_summary(summ_vals, epoch+30)
        weights_save["Compression" + str(epoch)] = sess.run(flat_weights)
        plt.scatter(weights_save["Pre Compression"], weights_save["Compression" + str(epoch)])

weights_save["Post Compression"] = sess.run(flat_weights)
plt.scatter(weights_save["Pre Compression"], weights_save["Post Compression"])
sess.close()

sess.run(priors["mu"])

tensorboard --logdir="C:\Users\bradley.tjandra\Documents\AI\SoftWeightSharing\Training"