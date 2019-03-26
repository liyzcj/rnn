# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:53:27 2019

@author: liyz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# =============================================================================
# Utilities for parsing PTB text files.
# =============================================================================


#import collections
import os
#import sys

import tensorflow as tf

def load_raw_data(data_path=None):
    
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
  
    with tf.gfile.GFile(valid_path, "r") as f:
        valid = f.read().replace("\n", "<eos>\n").split(sep='\n')
    
    train = tf.data.TextLineDataset(train_path)
    valid = tf.data.Dataset.from_tensor_slices(valid)
    test = tf.data.TextLineDataset(test_path)
    
    return train, valid, test


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    
    pass
  

def _parse_data(line):
    
    line_split = tf.string_split([line]).values
    index = words.lookup(line_split)
    
    return index


if __name__ == '__main__':
    
    train, valid, test = load_raw_data('data/')
    
    words = tf.contrib.lookup.index_table_from_file("vocab.txt")
    
    valid = valid.map(_parse_data)
    id_pad = words.lookup(tf.constant('<pad>'))
    valid = valid.padded_batch(3, padded_shapes=tf.TensorShape([None]), padding_values=id_pad)
        
    
    
    valid_iter = valid.make_initializable_iterator()
    next_valid = valid_iter.get_next()
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    
    with tf.Session() as sess:
        count = 0
        
#        sess.run(words.initializer)
        sess.run(variable_init_op)
        sess.run(valid_iter.initializer)
        
        try:
            print(sess.run(id_pad))
            while(True):
                v = sess.run(next_valid)
                print(v)
                count += 1
                if count > 2:
                    raise RuntimeError('stop')
        except tf.errors.OutOfRangeError:
            print(count)
        except RuntimeError as e:
            print(e)