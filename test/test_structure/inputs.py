from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np



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


class SEMInputs(Inputs):

    def __init__(self, file_path, params):
        print("instanciate a SEMInput!")
        data = self._data_preprocess(file_path)
        print("load success!")
        self._batches_one_epoch = (data.shape[0] // params.batch_size -1) // params.time_steps
        self._generator = self._batch_generater(data, params)

    def _data_preprocess(self, file_path):
        return np.load(file_path)

    def _batch_generater(self, data, params):
        """Generate one batch of inputs and targets"""

        # batch_size and time_steps
        bs = params.batch_size
        ts = params.time_steps
        # Batch length
        bl = data.shape[0] // params.batch_size
        data = data[:bs*bl, :].reshape(bs, bl, -1)
        # Compute how much batches in one epoch
        # (batch_size - 1) because targets is late one timestep
        self._batches_one_epoch = (bl-1) // ts
        # Generate inputs and targets
        while(True):
            np.random.shuffle(data)
            for i in range(self._batches_one_epoch):
                x = data[:, i*ts:(i+1)*ts]
                y = data[:, i*ts+1:(i+1)*ts+1]
                yield x, y
    @property
    def next_batch(self):
        return  self._generator.__next__()

    @property
    def boe(self):
        return self._batches_one_epoch
