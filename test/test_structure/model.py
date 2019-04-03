from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf
import time


class Model(metaclass=ABCMeta):

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def run_one_epoch(self):
        pass

    @abstractproperty
    def loss(self):
        pass



class SEMModel(Model):

    def __init__(self, params, inputs, is_training=False):
        
        print("Start Instantiate SEMModel...")
        self._is_training = is_training
        self._inputs = inputs
        # Items in one batch
        self.iib = params.batch_size * params.time_steps
        self._build_model(params)
        print("Instantiate SEMModel success!")

    def _build_model(self, params):
        self._build_inputs(params)
        lstm_outputs = self._build_lstm(params)
        logits = self._build_fnlayer(lstm_outputs, params)
        self._build_loss(logits)
        if not self._is_training:
            return

        self._build_optimizer(params)
        
        print("Build Compute Graph Success!")

    def _build_inputs(self, params):
        """Inputs of the model"""
        with tf.name_scope("Inputs"):
            
            self._x = tf.placeholder(
                    tf.float32, 
                    shape=(params.batch_size, 
                           params.time_steps, 
                           params.feature_dim),
                    name="Inputs")
            self._y = tf.placeholder(
                    tf.float32,
                    shape=(params.batch_size,
                           params.time_steps,
                           params.feature_dim),
                    name="Targets")

    def _build_lstm(self, params):
        """Build LSTM Layers"""
        with tf.name_scope("LSTM"):
            
            def lstm_cell():
                return tf.nn.rnn_cell.LSTMCell(params.hidden_units)
            
            def drop_cell():
                return tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell(), output_keep_prob=params.keep_prob)
                
            # IF Dropout
            cell = lstm_cell
            if self._is_training and params.keep_prob < 1:
                self._x = tf.nn.dropout(self._x, params.keep_prob)
                cell = drop_cell
            # For multi-layers and Instanciate cells
            stack_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [cell() for _ in range(params.num_layers)])
            # Initializtion of hidden state, Reture the tensors of hiddenstate
            self._initial_state = stack_cell.zero_state(params.batch_size, tf.float32)
            # Use dynamic rnn unpack in time dimention
            outputs, self._final_state = tf.nn.dynamic_rnn(
                    stack_cell,
                    self._x,
                    initial_state = self._initial_state,
                    )
            return outputs

    def _build_fnlayer(self, inputs, params):
        with tf.name_scope("FNLayer"):
            fn_w = tf.get_variable(
                    name="fnlayer_w",
                    shape=[params.hidden_units, params.feature_dim],
                    initializer=tf.initializers.truncated_normal(stddev=0.1))
            fn_b = tf.get_variable(
                    name="fnlayer_b",
                    shape=[params.feature_dim],
                    initializer=tf.initializers.zeros())
            # Compute logits
            inputs = tf.reshape(inputs, shape=[-1, params.hidden_units])
            logits = tf.matmul(inputs, fn_w) + fn_b
            logits = tf.reshape(logits, shape=[params.batch_size, params.time_steps, -1])
            return logits

    def _build_loss(self, logits):
        with tf.name_scope("Loss"):
            loss = tf.losses.mean_squared_error(self._y, logits)
            self._loss = tf.reduce_mean(loss)


    def _build_optimizer(self, params):
        with tf.name_scope("Optimizer"):
            self._lr = tf.Variable(0.0, trainable=False, name = "Learning_rate")
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          params.grad_clip,
                                          name="Gradient_Clip")

            # Optimizer
            optimizer = tf.train.GradientDescentOptimizer(self._lr)

            self.global_step = tf.train.get_or_create_global_step()
            # Train operation
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=self.global_step)
            
            self._new_lr = tf.placeholder(
                tf.float32,
                shape=[],
                name="new_learning_rate")
            # lr update operation
            self._lr_update = tf.assign(self._lr, self._new_lr)

    def run_one_epoch(self, sess):
        start_time = time.time()
        total_loss = 0.0
        state = sess.run(self._initial_state)

        # Fetch Dictionary
        fetches = {
                "loss": self._loss,
                "final_state": self._final_state
                }
        if self._is_training:
            fetches["train_op"] = self._train_op
        
        boe = self._inputs.boe
    
        for step in range(boe):
            x, y = self._inputs.next_batch
            feed = {
                self._x: x,
                self._y: y,
                self._initial_state: state
                }

            fetch = sess.run(fetches, feed_dict=feed)
            loss = fetch["loss"]
            state = fetch["final_state"]

            total_loss += loss
            if self._is_training and step % (boe // 10) == 0:
                print(f"{step}/{boe} Step Loss: "+ "%.3f  Speed: %.0f ips" % (loss*1000, (step+1) * self.iib / (time.time() - start_time)))

        return total_loss
    # Method to assign learning rate
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr:lr_value})


    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr