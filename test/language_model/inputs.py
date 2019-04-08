from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import tensorflow as tf



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

    def __init__(self, file_path, params, words):
        self.params = params
        print("instanciate a PTBInputs!")
        data = self._data_preprocess(file_path, words)
        self._next_batch = self._batch_generater(data)
        # print(data.shape)
        # print(data[-10:])
        print("load success!")
        # self._batches_one_epoch = (data.shape[0] // params.batch_size -1) // params.time_steps
        # self._generator = self._batch_generater(data, params)

    def _data_preprocess(self, file_path, words):
        bs = self.params.batch_size
        ts = self.params.time_steps
        with tf.gfile.GFile(file_path, "r") as f:
            data = f.read().replace("\n", "<eos>").split()
            remain = len(data) % bs
            if remain != 0:
                data = data[:-remain]
            data = np.array(data).reshape(bs, -1)
            data = np.transpose(data)
            x = data[:-1]
            self._batches_one_epoch = x.shape[0] // ts
            y = data[1:]
            x = tf.data.Dataset.from_tensor_slices(x)
            y = tf.data.Dataset.from_tensor_slices(y)

            x = x.map(lambda tokens: (words.lookup(tokens)))
            y = y.map(lambda tokens: (words.lookup(tokens)))
            data = tf.data.Dataset.zip((x,y))
            
        return data

    def _batch_generater(self, data):
        """Generate one batch of inputs and targets"""
        data = data.batch(self.params.time_steps, drop_remainder=True)
        data = data.repeat()
        it = data.make_initializable_iterator()
        self.init_op = it.initializer

        x, y = it.get_next()
        next_elem = {"inputs":x, "targets":y}
        return next_elem

    @property
    def next_batch(self):
        return self._next_batch
        # return  self._generator.__next__()

    @property
    def boe(self):
        return self._batches_one_epoch
