# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 08:59:26 2019

@author: liyz
"""
#%%
# =============================================================================
# Import
# =============================================================================
import tensorflow as tf
import numpy as np
#import sys

#%%
# =============================================================================
# dataset.repeat 是重复之前的操作
# =============================================================================
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10, dtype='float32'))
dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
#dataset = dataset.batch(25)
#dataset = dataset.shuffle(buffer_size=5)
#dataset = dataset.repeat(5)
dataset = dataset.shuffle(buffer_size=10).batch(4).repeat(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
        
    
#%%
# =============================================================================
# Initializable iterator
# =============================================================================

limit = tf.placeholder(dtype=tf.int32, shape=[])

dataset = tf.data.Dataset.from_tensor_slices(tf.range(start=0, limit=limit))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={limit: 15})
    try:
        while(True):
          value = sess.run(next_element)
          print(value)
    except tf.errors.OutOfRangeError:
        print("end!")
        
#%%
# =============================================================================
# dataset.repeat 是重复之前的操作
# =============================================================================
dataset = tf.data.TextLineDataset('test.txt')
#dataset = dataset.map(lambda x: x + 1) # 2.0, 3.0, 4.0, 5.0, 6.0
dataset = dataset.batch(2)
#dataset = dataset.shuffle(buffer_size=5)
#dataset = dataset.repeat(5)
#dataset = dataset.shuffle(buffer_size=10).batch(4).repeat(2)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")