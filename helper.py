# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 07:03:45 2018

@author: Bradley.Tjandra
"""

import sys
import tensorflow as tf
import numpy as np

def pretty_list(my_array, f):
    a = ", ".join(f.format(d) for d in my_array.tolist())
    return "[" + a +  "]" 

def print_prog_bar(message, step, n_steps, complete_message = None):
    
    if step % 5 != 0:
        return 0
    
    PROGRESS_BAR_LENGTH = 30
    sys.stdout.write("\r")
    if complete_message == None:
        complete_message = message
        
    if step >= n_steps:
        prog_bar = "=" * PROGRESS_BAR_LENGTH 
        sys.stdout.write("{}. Progress {} of {} ({:d}%): [{}]  \n"
                         .format(complete_message, step, n_steps, 100, prog_bar))
        sys.stdout.flush()
    else:
        perc_length = step / n_steps * 100
        prog_bar_length = int(perc_length / 100 * PROGRESS_BAR_LENGTH)
        prog_bar = "=" * prog_bar_length + (PROGRESS_BAR_LENGTH - prog_bar_length) * " "
        sys.stdout.write("{}. Progress {} of {} ({:d}%): [{}]  \r"
                         .format(message, step, n_steps, int(perc_length), prog_bar))
        sys.stdout.flush()

def shuffle_in_unison(arrays, index = 0, seed = 1):
    
    tf.set_random_seed(seed)
    if len(arrays) == 0:
        return []
    
    
    shuffle_range = arrays[0].shape[index]
    shuffled_arrays = []
    for arr in arrays:
        assert(arr.shape[index]==shuffle_range)
        shuffled_arr = np.empty(arr.shape, dtype=arr.dtype)
        shuffled_arrays.append(shuffled_arr)
    
    permutation = np.random.permutation(shuffle_range)
    for old, new in enumerate(permutation):
        for i in range(len(arrays)):
            shuffled_arrays[i][new] = arrays[i][old]
            
    return shuffled_arrays

def get_minibatch(arrays, batch_size, batch_number):
    
    result = []
    batch_start = batch_size * batch_number 
    batch_end = batch_size * batch_number + batch_size
    
    for arr in arrays:
        result.append(arr[batch_start:batch_end])
        
    return result