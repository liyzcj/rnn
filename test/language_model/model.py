from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf
import time
# from tensorflow.python.client import timeline

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



class PTBModel(Model):

    def __init__(self, params, inputs, is_training=False):
        
        print("Start Instantiate PTBModel...")
        self.params = params
        self._is_training = is_training
        self._inputs = inputs
        self._build_model()
        print("Instantiate PTBModel success!")

    def _build_model(self):
        inputs = self._build_inputs()
        lstm_outputs = self._build_lstm(inputs)
        logits = self._build_softmax(lstm_outputs)
        self._build_loss(logits)
        if not self._is_training:
            return

        self._build_optimizer()
        
        print("Build Compute Graph Success!")

    def _build_inputs(self):
        """Inputs of the model
        self._inputs: The object of inputs contains property 'next_batch'
        'next_batch' will return a dictionary contains 'inputs' and 'targets'..
        """
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
        with tf.name_scope("Inputs"):
            
            batch = self._inputs.next_batch
            self._x = batch['inputs']
            self._y = batch['targets']
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable(
                    "Embedding", 
                    [vocab_size, embedding_size],
                    dtype=tf.float32
                )
                inputs = tf.nn.embedding_lookup(self.embedding, self._x)
        return inputs

    def _build_lstm(self, inputs):
        """Build LSTM Layers"""
        # Hyper params
        keep_prob = self.params.keep_prob
        hidden_units = self.params.hidden_units
        is_training = self._is_training
        num_layers = self.params.num_layers
        batch_size = self.params.batch_size
        
        with tf.name_scope("LSTM"):
            
            def lstm_cell():
                return tf.nn.rnn_cell.LSTMCell(hidden_units)
            
            def drop_cell():
                return tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell(), output_keep_prob=keep_prob)
                
            # IF Dropout
            cell = lstm_cell
            if is_training and keep_prob < 1:
                inputs = tf.nn.dropout(inputs, keep_prob)
                cell = drop_cell
            # For multi-layers and Instanciate cells
            stack_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [cell() for _ in range(num_layers)])
            # Initializtion of hidden state, Reture the tensors of hiddenstate
            self._initial_state = stack_cell.zero_state(batch_size, tf.float32)
            # Use dynamic rnn unpack in time dimention
            outputs, self._final_state = tf.nn.dynamic_rnn(
                    stack_cell,
                    inputs,
                    time_major=True,
                    initial_state = self._initial_state,
                    )
            return outputs

    def _build_softmax(self, inputs):

        hidden_units = self.params.hidden_units
        vocab_size = self.params.vocab_size

        with tf.name_scope("Softmax"):
            softmax_w = tf.get_variable(
                    name="softmax_w",
                    shape=[hidden_units, vocab_size],
                    initializer=tf.initializers.truncated_normal(stddev=0.1))
            softmax_b = tf.get_variable(
                    name="softmax_b",
                    shape=[vocab_size],
                    initializer=tf.initializers.zeros())
            # Compute logits
            inputs = tf.reshape(inputs, shape=[-1, hidden_units])
            logits = tf.matmul(inputs, softmax_w) + softmax_b
            return logits

    def _build_loss(self, logits):
        bs = self.params.batch_size
        ts = self.params.time_steps
        with tf.name_scope("Loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._y, [-1])],
                [tf.ones([bs * ts], dtype = tf.float32)])
                
            self._loss = tf.reduce_sum(loss) / bs

    def _build_optimizer(self):
        grad_clip = self.params.grad_clip
        with tf.name_scope("Optimizer"):
            self._lr = tf.Variable(0.0, trainable=False, name = "Learning_rate")
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                grad_clip,
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
        iters = 0
        state = sess.run(self._initial_state)

        # Fetch Dictionary
        fetches = {
                "loss": self._loss,
                "final_state": self._final_state
                }
        if self._is_training:
            fetches["train_op"] = self._train_op
        
        boe = self._inputs.boe
        # ##############
        # run_metadata = tf.RunMetadata()
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # #############
        for step in range(boe):
            # start_time = time.time()
            feed = {self._initial_state: state}

            # fetch = sess.run(fetches, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            fetch = sess.run(fetches, feed_dict=feed)
            loss = fetch["loss"]
            state = fetch["final_state"]

            total_loss += loss
            iters += self.params.time_steps
            if self._is_training and step % (boe // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / boe, np.exp(total_loss / iters),
                #    400 / (time.time() - start_time)))
                   iters * self.params.batch_size / (time.time() - start_time)))
            # ######################
            # # Create the Timeline object, and write it to a json
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_mine_step_%d.json' % step, 'w') as f:
            #     f.write(chrome_trace)
            # #####################
        return np.exp(total_loss / iters)
    # Method to assign learning rate
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr:lr_value})


    @property
    def loss(self):
        return self._loss

    @property
    def lr(self):
        return self._lr