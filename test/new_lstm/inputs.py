from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf
import os



class Inputs(metaclass=ABCMeta):

    @abstractmethod
    def _data_preprocess(self):
        """ Preprocess data """
        pass


    @abstractmethod
    def _batch_generater(self):
        """ Generate one batch of x, y """
        pass

    @abstractproperty
    def next_batch(self):
        """ one batch of input data """
        pass
    

    @abstractproperty
    def boe(self):
        """ batches in one epoch """
        pass


class PTBInputs(Inputs):

    def __init__(self, file_path, params, name="PTBGenerator"):
        print("instanciate a PTBInputs!")
        self.params = params
        vocab_path = "test/new_lstm/vocab.txt"
        if not os.path.exists(vocab_path):
            raise Exception("Error! Don't have vocabulary file. run <build_ptb_vocab.py>")
        self.word2id = self._load_vocab(vocab_path)
        
        data = self._data_preprocess(file_path)
        self._batches_one_epoch = ((len(data) // params.batch_size) - 1) // params.time_steps
        self._next_batch = self._batch_generater(data, name)
        # print(data.shape)
        # print(data[-10:])
        print("load success!")
        # self._batches_one_epoch = (data.shape[0] // params.batch_size -1) // params.time_steps
        # self._generator = self._batch_generater(data, params)

    def _data_preprocess(self, file_path):
        with tf.gfile.GFile(file_path, "r") as f:
            data = f.read().replace("\n", "<eos>").split()
        data = [self.word2id[word] for word in data if word in self.word2id]
        return data


    def _load_vocab(self, vocab_path):

        with open(vocab_path) as f:
            word2id = eval(f.read())
        return word2id

    def _batch_generater(self, data, name):
        """Generate one batch of inputs and targets"""
        bs = self.params.batch_size
        ts = self.params.time_steps
        with tf.name_scope(name,values=[data, bs, ts]):
            data = tf.convert_to_tensor(data, name="Raw_data", dtype=tf.int32)
            # length of total data
            data_len = tf.size(data)
            # length of one batch
            batch_len = data_len // bs
            # reshape to [batch_size, batch_len]
            data = tf.reshape(data[:bs * batch_len],[bs, batch_len])
            # batches in one epoch
            boe = (batch_len - 1) // ts
            # make sure boe is not 0
            assertion = tf.assert_positive(
                boe,
                message="Batches in one epoch == 0, decrease batch_size or time_steps!")
            with tf.control_dependencies([assertion]):
                boe = tf.identity(boe, name="Batches_one_epoch")
            # range input producer
            i = tf.train.range_input_producer(boe, shuffle=False).dequeue()
            # slice one in put from raw tensor
            x = tf.strided_slice(data, [0, i * ts], [bs, (i+1) * ts])
            x.set_shape([bs,ts])
            y = tf.strided_slice(data, [0, i * ts + 1], [bs, (i+1)*ts+1])
            y.set_shape([bs,ts])
            return x,y
    @property
    def next_batch(self):
        return self._next_batch
        # return  self._generator.__next__()

    @property
    def boe(self):
        return self._batches_one_epoch