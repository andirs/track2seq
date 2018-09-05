import math
import numpy as np
import os
from tools.io import load_obj, store_obj
from abc import ABC, abstractmethod

class BatchGenerator(ABC):
    """
    Abstract generator class to create batches of training data. 
    Turns data stream into x and y batches to train Track2Seq model.
    
    Use SimpleBatchGenerator or FilterPrecicionBatchGenerator for 
    batch generation.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def calc_steps_per_epoch(self):
        pass

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def store_step_counter(self, s):
        pass

    @abstractmethod
    def store_epoch_counter(self, e):
        pass


class SimpleBatchGenerator(BatchGenerator):
    def __init__(self, data, seq_length, n_batch_size, step=5, store_folder='step_point/'):
        """
        Batch generator that turns long sequence into x and y training
        examples. Naive implementation where end-of-string sequences
        are treated as normal words.
        
        Parameters:
        --------------
        data:           list, can be either training, validation or test data
        seq_length:     int, number of tracks that will be fed into the network
        n_batch_size:   int, amount of trainings examples per batch
        step:           int, number of words to be skipped over between training samples within each batch
        store_folder:   str, folder name where checkpoint and step info will be stored
        
        Returns:
        -------------
        None
        """
        self.data = data
        self.seq_length = seq_length
        self.n_batch_size = n_batch_size
        self.store_folder = store_folder
        self.step = step

        if not os.path.exists(self.store_folder):
            os.makedirs(self.store_folder)

        # current_idx will save progress and serve as pointer
        # will reset to 0 once end is reached
        if os.path.exists(os.path.join(self.store_folder, 'global_step_point.pckl')):
            self.current_idx = load_obj(os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')
        else:
            self.current_idx = 0

        self.steps_per_epoch = self.calc_steps_per_epoch()

        # reload or initialize epoch and step counter
        if os.path.exists(os.path.join(self.store_folder, 'global_epoch_point.pckl')):
            self.epoch_counter = load_obj(os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')
        else:
            self.epoch_counter = 0

    def calc_steps_per_epoch(self):
        """
        Calculates total steps per epoch based on parameters.
        
        Paramters:
        -------------
        None
        
        Returns:
        -------------
        int, total steps per epoch
        """
        # calculate steps per epoch
        return (len(self.data) // self.n_batch_size - 1) // self.step

    def store_step_counter(self, s):
        """
        Stores step counter on harddrive.
        
        Paramters:
        -------------
        s:      int, number of steps
        
        Returns:
        -------------
        None
        """
        store_obj(s, os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')

    def store_epoch_counter(self, e):
        """
        Method to store epoch counter. 
        Updates internal epoch counter as well.

        Parameters:
        --------------
        e: int, epoch count

        Returns:
        --------------
        None
        """
        self.epoch_counter = e
        store_obj(self.epoch_counter, os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')

    def generate(self):
        """
        Generator function that yields training data.

        Yields:
        --------------
        x, y: np.array, np.array: training sequence and training labels
        """
        x = np.zeros((self.n_batch_size, self.seq_length))
        y = np.zeros((self.n_batch_size, self.seq_length))
        while True:
            for i in range(self.n_batch_size):
                if self.current_idx + self.seq_length >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0

                x[i, :] = self.data[self.current_idx:self.current_idx + self.seq_length]
                y[i, :] = self.data[self.current_idx + 1:self.current_idx + self.seq_length + 1]
                self.current_idx += self.step
            yield x, y


class FilterPrecicionBatchGenerator(BatchGenerator):
    """
    Batch generator that turns long sequence into x and y training
    examples. Generator `splits` sequence on eos char and generates
    training examples. 

    Parameters:
    --------------
    data:               list, can be either training, validation or test data
    seq_length:         int, number of tracks that will be fed into the network
    n_batch_size:       int, amount of trainings examples per batch
    eos_char:           int, end-of-sequence-token id in vocabulary
    unk_char:           int, unknown-token id in vocabulary
    remove_unk_chars:   bool, utility flag to remove all unknown tokens from data  
    store_folder:       str, folder name where checkpoint and step info will be stored

    Returns:
    -------------
    None
    """
    def __init__(self, data, seq_length, n_batch_size=4, eos_char=0, unk_char=1, remove_unk_chars=False,
                 store_folder='step_point/'):
        self.data = data
        self.n_batch_size = n_batch_size
        self.seq_length = seq_length
        self.length = 0
        self.eos_char = eos_char
        self.steps_per_epoch = self.calc_steps_per_epoch()
        self.store_folder = store_folder

        if remove_unk_chars:
            self.data = [x for x in self.data if x != unk_char]

        if not os.path.exists(self.store_folder):
            os.makedirs(self.store_folder)

        if os.path.exists(os.path.join(self.store_folder, 'global_step_point.pckl')):
            self.current_idx = load_obj(os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')
        else:
            self.current_idx = 0

        # reload or initialize epoch and step counter
        if os.path.exists(os.path.join(self.store_folder, 'global_epoch_point.pckl')):
            self.epoch_counter = load_obj(os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')
        else:
            self.epoch_counter = 0

        # for generator to be called once
        self.single_batch_sequence_pointer = self.current_idx
        self.x_pointer, self.y_pointer, = self.current_idx, self.current_idx + 1

    def store_step_counter(self, s):
        """
        Stores step counter on harddrive.

        Paramters:
        -------------
        s:      int, number of steps

        Returns:
        -------------
        None
        """
        store_obj(s, os.path.join(self.store_folder, 'global_step_point.pckl'), 'pickle')

    def store_epoch_counter(self, e):
        """
        Method to store epoch counter. 
        Updates internal epoch counter as well.

        Parameters:
        --------------
        e: int, epoch count

        Returns:
        --------------
        None
        """
        self.epoch_counter = e
        store_obj(self.epoch_counter, os.path.join(self.store_folder, 'global_epoch_point.pckl'), 'pickle')

    def generate(self):
        """
        Generator function that yields training data.

        Yields:
        --------------
        x, y: np.array, np.array: training sequence and training labels
        """
        x_sequence, y_sequence = [], []
        batch_item = 0
        end_of_sequence = False
        x = np.zeros((self.n_batch_size, self.seq_length))
        y = np.zeros((self.n_batch_size, self.seq_length))
        while True:
            while batch_item < self.n_batch_size:
                self.x_pointer = self.single_batch_sequence_pointer
                self.y_pointer = self.x_pointer + 1
                while len(x_sequence) < self.seq_length:
                    if self.x_pointer >= len(self.data) or self.y_pointer == len(self.data):
                        self.single_batch_sequence_pointer = 0
                        end_of_sequence = True
                        self.current_idx = 0
                        break
                    x_candidate = self.data[self.x_pointer]
                    y_candidate = self.data[self.y_pointer]
                    if y_candidate == self.eos_char:
                        self.single_batch_sequence_pointer = self.y_pointer
                        x_sequence, y_sequence = [], []
                        break
                    else:
                        x_sequence.append(x_candidate)
                        y_sequence.append(y_candidate)
                        self.x_pointer += 1
                        self.y_pointer += 1
                if not end_of_sequence:
                    self.single_batch_sequence_pointer += 1
                else:
                    end_of_sequence = False
                if x_sequence:
                    x[batch_item, :] = x_sequence
                    y[batch_item, :] = y_sequence
                    batch_item += 1
                x_sequence, y_sequence = [], []
            self.current_idx += 1
            yield x, y

    def calc_steps_per_epoch(self):
        """
        Calculates total steps per epoch based on parameters.

        Paramters:
        -------------
        None

        Returns:
        -------------
        int, total steps per epoch
        """
        total_count = 0
        tmp_seq_count = 0
        for el in self.data:
            if el == self.eos_char:
                total_count += (tmp_seq_count - self.seq_length)
                tmp_seq_count = 0
            else:
                tmp_seq_count += 1
        return math.ceil(total_count / self.n_batch_size)

