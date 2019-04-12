from abc import ABCMeta, abstractmethod, abstractproperty
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


class RAEInputs(Inputs):

    def __init__(self, file_path, params, vocab):
        data = self._data_preprocess(file_path, vocab)
        self._next_batch = self._batch_generater(data)

    def _data_preprocess(self, file_path, vocab):
        """Create tf.data Instance from txt file

        Args:
            path_txt: (string) path containing one example per line
            vocab: (tf.lookuptable)

        Returns:
            dataset: (tf.Dataset) yielding list of ids of tokens for each example
        """
        with tf.gfile.GFile(file_path, "r") as f:
            data = f.read().replace("\n", "<eos>\n").split(sep='\n')
        # How many batch in on epoch
        self._batches_one_epoch = len(data)

        data = tf.data.Dataset.from_tensor_slices(data)

        # Convert line into list of tokens, splitting by white space
        data = data.map(lambda string: tf.string_split([string]).values)

        # Lookup tokens to return their ids
        data = data.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))
        
        return data

    def _batch_generater(self, data):
        """Input function for NER

        Args:
            mode: (string) 'train', 'valid' or any other mode you can think of
                        At training, we shuffle the data and have multiple epochs
            sentences: (tf.Dataset) yielding list of ids of words
            datasets: (tf.Dataset) yielding list of ids of tags
            params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

        """
        # make sure you always have one batch ready to serve
        # data = data.repeat().prefetch(1)

        # Create initializable iterator from this dataset so that we can reset at each epoch
        iterator = data.make_initializable_iterator()

        # Query the output of the iterator for input to the model
        (sentence, length) = iterator.get_next()
        # Initialize operation of dataset operator
        self.init_op = iterator.initializer

        # Build and return a dictionnary containing the nodes / ops
        batch = {
            'sentence': sentence,
            'length': length
        }
        return batch


    @property
    def next_batch(self):
        return self._next_batch
        # return  self._generator.__next__()

    @property
    def boe(self):
        return self._batches_one_epoch
