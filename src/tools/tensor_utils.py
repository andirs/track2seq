import tensorflow as tf

class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    """
    Helper class if LSTM layers should be divided
    on multiple GPUs.
    """
    def __init__(self, device, cell):
        self._cell = cell
        self._device = device

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope)