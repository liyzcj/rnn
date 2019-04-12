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



class RAEModel(Model):

    def __init__(self, params, inputs):
        
        print("Start Instantiate PTBModel...")
        self.params = params
        self._inputs = inputs
        self._build_model()
        print("Instantiate PTBModel success!")

    def _build_model(self):
        embeded = self._build_inputs()
        reconstruct = self._encoder_decoder(embeded)
        logits = self._build_softmax(reconstruct)
        self._build_loss(logits)
        self._summries()
        self._build_optimizer()
        
        print("Build Compute Graph Success!")

    def _build_inputs(self):
        """Inputs of the model
        self._inputs: The object of inputs contains property 'next_batch'
        'next_batch' will return a dictionary contains 'sentence' and 'length'..
        """
        vocab_size = self.params.vocab_size
        embedding_size = self.params.embedding_size
         
        batch = self._inputs.next_batch
        self._sentence = tf.reshape(batch['sentence'], [1,-1], "Sentence")
        self._length = batch['length']

        with tf.name_scope("Embedding"):
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable(
                    "Embedding",
                    [vocab_size, embedding_size],
                    dtype=tf.float32
                )
                embeded = tf.nn.embedding_lookup(self.embedding, self._sentence)
        return embeded

    def _encoder_decoder(self, inputs):
        """Build Encoder"""
        kp = self.params.keep_prob
        hu = self.params.hidden_units
        nl = self.params.num_layers
        ts = self._length
        
        def lstm_cell():
            return tf.nn.rnn_cell.LSTMCell(hu)
        def drop_cell():
            return tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell(), output_keep_prob=kp
            )
        cell = lstm_cell
        if kp < 1:
            inputs = tf.nn.dropout(inputs, kp)
            cell = drop_cell
        with tf.name_scope("Autoencoder"):
            with tf.variable_scope("Encoder"):
                encoder = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(nl)])
                # Initialization of hidden state, Reture the tensors of hiddenstate
                encoder_initial_state = encoder.zero_state(1, tf.float32)
                # state = encoder_initial_state
                # for t in range(tf.cast(ts, tf.float32)):
                #     if t > 0: tf.get_variable_scope().reuse_variables()
                #     (output, state) = encoder(inputs[:,t,:], state)
                
                # Use dynamic rnn unpack in time dimention
                _, state = tf.nn.dynamic_rnn(
                        encoder,
                        inputs,
                        initial_state = encoder_initial_state,
                        )
            state = tf.reshape(tf.strided_slice(state,[0,1,0,0],[1,2,1,200]), [1,1,200])
            self._encoded = tf.tile(state, multiples=[1, ts, 1])
            with tf.variable_scope("Decoder"):
                decoder = tf.nn.rnn_cell.MultiRNNCell([cell() for _ in range(nl)])
                # Initialization of hidden state
                decoder_initial_state = decoder.zero_state(1, tf.float32)
                # state = decoder_initial_state
                # outputs = []
                # for t in range(tf.cast(ts, tf.float32)):
                #     if t > 0: tf.get_variable_scope().reuse_variables()
                #     output, state = decoder(self._encoded, state)
                #     outputs.append(output)
                outputs, _ = tf.nn.dynamic_rnn(
                        decoder,
                        self._encoded,
                        initial_state = decoder_initial_state,
                        )
                reconstruct = tf.reshape(outputs, [-1, hu], name="Reconstruction")
        return reconstruct

    def _build_softmax(self, inputs):

        hu = self.params.hidden_units
        vs = self.params.vocab_size

        with tf.name_scope("Softmax"):
            softmax_w = tf.get_variable(
                    name="softmax_w",
                    shape=[hu, vs],
                    initializer=tf.initializers.truncated_normal(stddev=0.1))
            softmax_b = tf.get_variable(
                    name="softmax_b",
                    shape=[vs],
                    initializer=tf.initializers.zeros())
            # Compute logits
            logits = tf.matmul(inputs, softmax_w) + softmax_b
            return logits

    def _build_loss(self, logits):
        ts = self._length
        with tf.name_scope("Loss"):
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [logits],
                [tf.reshape(self._sentence, [-1])],
                [tf.ones([ts], dtype = tf.float32)])
                
            self._loss = tf.reduce_sum(loss)

    def _summries(self):
        tf.summary.scalar(name="Loss", tensor=self._loss)
        self._summ_op = tf.summary.merge_all()

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

    def run_one_epoch(self, sess, writer=None):
        start_time = time.time()
        total_loss = 0.0

        # Fetch Dictionary
        fetches = {"loss": self._loss}
        fetches["summaries"] = self._summ_op
        fetches["train_op"] = self._train_op
        
        boe = self._inputs.boe
        # ##############
        # run_metadata = tf.RunMetadata()
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # #############
        for step in range(boe):

            # fetch = sess.run(fetches, feed_dict=feed, options=run_options, run_metadata=run_metadata)
            fetch = sess.run(fetches)
            loss = fetch["loss"]
            summ = fetch["summaries"]
            writer.add_summary(summ, global_step=tf.train.global_step(sess, self.global_step))
            total_loss += loss
            if step % (boe // 10) == 10:
                print("%.3f speed: %.0f sentence - ps" %
                  (step * 1.0 / boe, (step+1) / (time.time() - start_time)))
            # ######################
            # # Create the Timeline object, and write it to a json
            # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            # chrome_trace = fetched_timeline.generate_chrome_trace_format()
            # with open('timeline_no_dynamic_step_%d.json' % step, 'w') as f:
            #     f.write(chrome_trace)
            # #####################
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