# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:13:37 2019

@author: liyz
"""


import tensorflow as tf
import numpy as np
#import sys
import os
import time




class Input:
    def __init__(self, data_path , params):
        raw_data = np.load(data_path)
        self.generator = self.input_generator(raw_data,
                                              batch_size=params.batch_size,
                                              time_step=params.time_steps)
        
    def input_generator(self, array, batch_size = 20, time_step = 20):
        """
        Params:
            array: Numpy array with shape (number of samples, number of dim)
            batch_size: Number of samples in one batch
            time_step: Number of sequences length
        Return:
            A generator, generator one batch with shape (batch_size, time_step, dim)
        """
        # Number of samples in each batch
        batch_length = array.shape[0] // batch_size
        array = array[:batch_size*batch_length,:].reshape(batch_size, batch_length, -1)
        # Calculate how much batches in one epoch
        self.steps_one_epoch = (batch_length-1) // time_step
        # Make sure the samples is enough
        if self.steps_one_epoch <= 1:
            raise Exception("samples is too less!")
        # Generate x and y batch
        while(True):
            np.random.shuffle(array)
            for i in range(self.steps_one_epoch):
                x = array[:, i*time_step:(i+1)*time_step]
                y = array[:, i*time_step+1:(i+1)*time_step+1]
                yield x, y
            
class Params:
    num_epoch = 10
    batch_size = 8
    time_steps = 5
    hidden_units = 200
    embedding_size = None
    feature_dim = 512
    vocab_size = None
    is_training = True
    num_layers = 2
    grad_clip = 5
    verbose = True
    restore_from = None
    lr_decay = 0.5
    decay_epoch = 5
    learninig_rate = 0.1
    init_scale = 0.1
    keep_prob = 1.0
    
class RnnModel:
    
    def __init__(self, params, is_training):
        
        self.is_training = is_training
        self._build_model(params)
        
        
    def _build_model(self, params):
        """
        Build Model.
        Params:
            params: Config for model
        """
        
# =============================================================================
# Inputs
# =============================================================================
        with tf.name_scope("Inputs"):
            
            self.inputs = tf.placeholder(
                    tf.float32, 
                    shape=(params.batch_size, 
                           params.time_steps, 
                           params.feature_dim),
                    name="inputs")
            self.targets = tf.placeholder(
                    tf.float32,
                    shape=(params.batch_size,
                           params.time_steps,
                           params.feature_dim),
                    name="targets")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            # Embedding if need
            lstm_inputs = self.inputs
            if params.embedding_size != None:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                            "embedding", 
                            [params.vocab_size, params.embedding_size])
                    lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
# =============================================================================
# LSTM layer
# =============================================================================
        with tf.name_scope("LSTM"):
            
            def lstm_cell():
                return tf.nn.rnn_cell.LSTMCell(params.hidden_units)
            
            def drop_cell():
                return tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell(), output_keep_prob=params.keep_prob)
                
            # IF Dropout
            cell = lstm_cell
            if self.is_training and params.keep_prob < 1:
                lstm_inputs = tf.nn.dropout(lstm_inputs, params.keep_prob)
                cell = drop_cell
            # For multi-layers and Instanciate cells
            stack_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [cell() for _ in range(params.num_layers)])
            # Initializtion of hidden state, Reture the tensors of hiddenstate
            self.initial_state = stack_cell.zero_state(params.batch_size, tf.float32)
            # Use dynamic rnn unpack in time dimention
            lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                    stack_cell,
                    lstm_inputs,
                    initial_state = self.initial_state,
                    )
            # The shape of outputs is [batch_size, time_steps, hidden_units]
#            # concatenate the lstm outputs
#            self.outputs = tf.reshape(
#                    tf.concat(lstm_outputs, 1, name="concat"),
#                    [-1, params.hidden_units],
#                    name="outputs")
#            # Softmax layer
# =============================================================================
# Softmax Layer
# =============================================================================
#        with tf.name_scope("Softmax"):
#            with tf.variable_scope("softmax"):
#                softmax_w = tf.get_variable(
#                        "softmax_w",
#                        shape=[params.hidden_units, params.vocab_size],
#                        initializer=tf.initializers.truncated_normal(stddev=0.1))
#                softmax_b = tf.get_variable(
#                        name="softmax_b",
#                        shape=[params.vocab_size],
#                        initializer=tf.initializers.zeros())
##                softmax_w = tf.Variable(
##                        tf.truncated_normal([params.hidden_units, params.vocab_size], 
##                                            stddev=0.1))
##                softmax_b = tf.Variable(tf.zeros(params.vocab_size))
#                
#            # compute logits
#            logits = tf.matmul(self.outputs, softmax_w) + softmax_b
#            self.predict_prob = tf.nn.softmax(logits, name="preditions")

# =============================================================================
# Full Connextion layers
# =============================================================================
        with tf.name_scope("FNLayer"):
            with tf.variable_scope("fnlayer"):
                softmax_w = tf.get_variable(
                        name="fnlayer_w",
                        shape=[params.hidden_units, params.feature_dim],
                        initializer=tf.initializers.truncated_normal(stddev=0.1))
                softmax_b = tf.get_variable(
                        name="fnlayer_b",
                        shape=[params.feature_dim],
                        initializer=tf.initializers.zeros())
#                softmax_w = tf.Variable(
#                        tf.truncated_normal([params.hidden_units, params.vocab_size], 
#                                            stddev=0.1))
#                softmax_b = tf.Variable(tf.zeros(params.vocab_size))
                
            # compute logits
            lstm_outputs = tf.reshape(lstm_outputs, shape=[-1, params.hidden_units])
            logits = tf.matmul(lstm_outputs, softmax_w) + softmax_b
            logits = tf.reshape(logits, shape=[params.batch_size, params.time_steps, -1])
            self.outputs = tf.nn.sigmoid(logits, name="sigmoid")
            
# =============================================================================
# Loss
# =============================================================================
        with tf.name_scope("Loss"):
            loss = tf.losses.mean_squared_error(self.targets,
                                                self.outputs)
            self.loss = tf.reduce_mean(loss)
# =============================================================================
# Optimizer
# =============================================================================
        if not self.is_training:
            return
        # Learning rate
        self._lr = tf.Variable(0.0, trainable=False, name = "lr")
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          params.grad_clip,
                                          name="Gradient_Clip")
        
        # Optimizer
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # Train operation
        self.train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(
                tf.float32,
                shape=[],
                name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        
        
        
    # Method to assign learning rate
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr:lr_value})
    
    @property
    def lr(self):
        return self._lr
    
def one_epoch(session, model, inputs, params, is_training=True):
    start_time = time.time()
    total_loss = 0.0
    step = 0
    state = session.run(model.initial_state)
    # Fetch Dictionary
    fetches = {
            "loss": model.loss,
            "final_state": model.final_state
            }
    if is_training:
        fetches["train_op"] = model.train_op
    
    for x,y in inputs.generator:
        step += 1
        
        feed_dict = {
                model.inputs: x,
                model.targets: y,
                model.initial_state: state
                }
        vals = session.run(fetches, feed_dict=feed_dict)
        loss = vals["loss"]
        state = vals["final_state"]
        
        total_loss += loss
#        print(step)
#        print(inputs.steps_one_epoch)
        if params.verbose and step % (inputs.steps_one_epoch // 10) == 1:
            print("%.3f Total cost: %.3f speed: %.0f wps" %
                  (step / inputs.steps_one_epoch, loss,
                   step * params.batch_size * params.time_steps / (time.time() - start_time)))
            
        if step == inputs.steps_one_epoch:
            break
    return total_loss
    
    
        
        
if __name__ == "__main__":
    
    data_path = "../final/data/ptb/"
    train_path = os.path.join(data_path, "ptb.train.txt.npy")
    valid_path = os.path.join(data_path, "ptb.valid.txt.npy")
    test_path = os.path.join(data_path, "ptb.test.txt.npy")
    
    params = Params()
    
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-params.init_scale,
                                                params.init_scale)
        
        with tf.name_scope("Train"):
            train_inputs = Input(train_path, params)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = RnnModel(params, is_training=True)
                
        with tf.name_scope("Valid"):
            valid_inputs = Input(valid_path, params)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = RnnModel(params, is_training=False)
        with tf.name_scope("Test"):
            test_inputs = Input(test_path, params)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = RnnModel(params, is_training=False)
                
        init = tf.global_variables_initializer()
    
        saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir="logs/", init_op=init)
        saver=sv.saver   
        
        with sv.managed_session() as sess:    
            for epoch in range(params.num_epoch):
                lr_decay = params.lr_decay ** max(epoch + 1 - params.decay_epoch, 0.0)
                train_model.assign_lr(sess, params.learninig_rate * lr_decay)
                print("Epoch: %d Learning rate: %.3f" % (epoch + 1, sess.run(train_model.lr)))
                train_loss = one_epoch(sess, train_model, train_inputs, params, is_training=True)
                print("Epoch: %d Train Loss: %.3f" % (epoch + 1, train_loss))
                valid_loss = one_epoch(sess, valid_model, valid_inputs, params, is_training=False)
                print("Epoch: %d Valid Perplexity: %.3f" % (epoch + 1, valid_loss))
            test_loss = one_epoch(sess, test_model, test_inputs, params, is_training=False)
            print("Test Loss : %.3f" % test_loss)