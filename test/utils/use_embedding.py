# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:26:50 2019

@author: liyz
"""

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os

def load_data(file_path):

    dataset = tf.data.TextLineDataset(file_path)
    
    it = dataset.make_one_shot_iterator()
    
    return it.get_next()


def embed_dataset_use(next_element):
    
    # load USE model
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
    # Import the USE module
    embed = hub.Module(module_url)

    embedding = embed([next_element])
    
    with tf.Session() as sess:
        
        print("global Initialization...")
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        print("Done")
        
        res = []
        count = 0
        try:
            while(True):
                embeded = sess.run(embedding)
                count += 1
                res.append(embeded)
                print("Number:", count)
                
        except tf.errors.OutOfRangeError:
            
            print("Embeding Finish!, Total:", count)
    print("Start Concatenating...")
    total_result = np.concatenate(res)
    print("Done")
    return total_result
    
def _save_numpy(array, filename):
    np.save(filename, array)
    
def embed_file(file_path):
    print("Start to embed file:" + file_path)
    next_elem = load_data(file_path)
    embeded = embed_dataset_use(next_elem)
    _save_numpy(embeded, file_path)
    print("Successful to embed " + file_path)
    print("Saved to: " + file_path + ".npy")
    
    
    
if __name__ == "__main__":
    
    
    data_path = "../data/ptb/"
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    train_path = os.path.join(data_path, "ptb.train.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    
    embed_file(train_path)
    embed_file(test_path)

