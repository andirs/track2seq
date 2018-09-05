import argparse
import math
import numpy as np
import os
import sys
import tensorflow as tf
import time

from collections import Counter
from tools.io import bool_parser, extract_pids, load_obj, store_obj, write_recommendations_to_file
from tools.batch_generators import SimpleBatchGenerator, FilterPrecicionBatchGenerator
from tools.tensor_utils import DeviceCellWrapper

print ('#' * 80)
print ('Track2Seq Model')
print ('#' * 80)

class Seq2Track(object):
    """
    Deep LSTM network class. Generates recommendation system using
    sequence continuation. 

    Parameters:
    --------------
    n_batch_size:   int, amount of observations per batch
    seq_length:     int, length of sequence
    n_vocab:        int, amount of tracks in vocabulary
    n_layers:       int, number of layers for network
    learning_rate:  float, learning rate for update rule
    latent_size:    float, size of embedding and layer
    recommendation: bool, if set to True, recommendation process starts
    training_type:  str, `full` or `nce` 
    """
    def __init__(self, n_batch_size, seq_length, n_vocab, n_layers, learning_rate=.001, latent_size=128, recommendation=False, training_type='full', teacher_forcing='full'):
        self.n_batch_size = n_batch_size
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.training_type = training_type
        self.teacher_forcing = teacher_forcing

        if recommendation:
            self.n_batch_size = 1
            self.seq_length = 1

        # define placeholders for X and y batches
        self.X = tf.placeholder(tf.int32, [None, self.seq_length], name='X')
        self.y = tf.placeholder(tf.int32, [None, self.seq_length], name='y')

        # generate embedding matrix for data representation and initialize randomly
        self.embedding_matrix = tf.get_variable('embedding_mat', [self.n_vocab, self.latent_size], tf.float32, tf.random_normal_initializer())
        self.embedding_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.X)

        # define an initial state for LSTM
        # since LSTM contain two states c and h we're working with the second dimension is 2
        self.initial_state = tf.placeholder(tf.float32, [self.n_layers, 2, self.n_batch_size, self.latent_size], name='initial_state')

        # states can be represented as tuples (c, h) per layer
        # to do so, we'll unstack the tensor on the layer axis
        state_list = tf.unstack(self.initial_state, axis=0)
        # and create a tuple representation for any (c, h) state representation per layer (n)
        # tuple(LSTMStateTuple(c0, h0), LSTMStateTuple(c1, h1), ..., LSTMStateTuple(cn, hn),)
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_list[i][0], state_list[i][1]) for i in range(self.n_layers)]
            )

        # in case one layer is being used
        cell = tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1.0)  # different size possible?

        #devices = ['/gpu:0', '/gpu:1']  # multi gpu layout - amount of devices ==  amount of layers
        def build_cells(layers, recommendation=recommendation, dropout_prob=.7):
            cells = []
            for i in range(layers):
                cell = tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True)
                #cell = DeviceCellWrapper(devices[i], tf.contrib.rnn.LSTMCell(self.latent_size, forget_bias=1., state_is_tuple=True))
                if not recommendation:
                    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_prob)
                cells.append(cell)
            return cells

        # otherwise create multirnn cells
        if self.n_layers > 1:
            cells = build_cells(self.n_layers, recommendation, 1.)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # generate state and y output per timestep
        self.output, self.state = tf.nn.dynamic_rnn(cell, self.embedding_inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # reshape so output fits into softmax function
        # [n_batch_size * seq_length, latent_size]
        self.output = tf.reshape(self.output, [-1, self.latent_size])

        # now we need to calculate the activations
        with tf.variable_scope('lstm_vars', reuse=tf.AUTO_REUSE):
            self.W = tf.get_variable('W', [self.latent_size, self.n_vocab], tf.float32, tf.random_normal_initializer())
            self.b = tf.get_variable('b', [self.n_vocab], tf.float32, tf.constant_initializer(0.0))
        self.logits = tf.matmul(self.output, self.W) + self.b

        # seq2seq.sequence_loss method requires [n_batch_size, seq_length, n_vocab] shaped vector
        self.logits = tf.reshape(self.logits, [self.n_batch_size, self.seq_length, self.n_vocab])

        # targets are expected to be of shape [seq_len, 1] where the second dimension represents the class as int
        # we can introduce weights regarding the tracks, this might be interesting for
        # an emulated attention mechanism or if we use artist / genre level recommendations
        # could also be used to weigh the first tracks or last tracks of a sequence 
        # with more importance
        if not recommendation and self.training_type == 'nce':
            W_t = tf.transpose(self.W)
            res_y = tf.reshape(self.y, [-1, 1])
            self.loss = tf.nn.nce_loss(
                weights=W_t, 
                biases=self.b,
                labels=res_y,
                inputs=self.output,
                num_sampled=400,
                num_true=1,
                num_classes=self.n_vocab)
            self.cost = tf.reduce_sum(self.loss) / (self.n_batch_size * self.seq_length)
        else:
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=self.y,
                weights=tf.ones([n_batch_size, seq_length], dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True)

            self.cost = tf.reduce_sum(self.loss)

        # accuracy calculations follow
        if recommendation:
            self.softmax = tf.nn.softmax(tf.reshape(self.logits, [-1, self.n_vocab]))
            self.predict = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)
            correct_predictions = tf.equal(self.predict, tf.reshape(self.y, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        self.lr = learning_rate
        with tf.variable_scope('lstm_vars', reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.training_op = self.optimizer.minimize(self.loss)
    
    def recommend(self, sess, start_sequence, int2track, track2int, n=100):
        """
        Recommendation method 
        Parameters:
        --------------
        sess:               tf.Session, TensorFlow session with initialized graph
        start_sequence:     list, list of spotify track uris 
        int2track:          dict, dictionary translating int representation to uri
        track2int:          dict, dictionary translating uri representation to int
        n:                  int, number of recommendations
        teacher_forcing:    str, 'full' or 'semi' flag for fully- or semi-guided teacher forcing

        Returns:
        --------------
        return_candidates: list, n recommendations in spotify uri format
        """
        def reduced_argsort(arr, size=n):
            """
            Optimized argsort method filtering for the top n values.
            Returns indices of highest values in descending order.

            Parameters:
            --------------
            arr:   list, list of floats or integers
            size:  int, maximum amount of values to return from array

            Returns:
            --------------
            values:  list, indices of highest values in descending order
            """
            return np.argpartition(arr, -size)[-size:][::-1]

        def artist_search(preds, candidates, int2track, seeds, n):
            """
            Adds candidate probabilities to candidates dictionary. 

            Parameters:
            --------------
            preds:        list, probabilities of predictions
            candidates:   dict, dictionary mapping from uri to probability sum
            int2track:    dict, dictionary mapping form int to uri
            seeds:        list, list of seed uris
            n:            int, number of probabilities to take into consideration

            Returns:
            ---------------
            index,        int, index of highest probability that's not eos or unknown token
            """
            samples = reduced_argsort(arr=preds, size=n)
            for sample in samples:
                track = int2track[sample]
                if track in seeds:
                    continue
                if track in candidates:
                    candidates[track] += preds[sample]
                else:
                    candidates[track] = preds[sample]

            # return index of highest probability
            pointer = 0

            # filter out eos and unknown token for stream of conciousness
            while int2track[samples[pointer]] in ['<eos>', 'unknown']:
                pointer += 1
            return samples[pointer]

        state = np.zeros(
            (self.n_layers, 2, self.n_batch_size, self.latent_size))

        candidates = {}

        # iterate over seeds and generate initial state for recommendation
        for track in start_sequence:
            x = np.zeros((1, 1))
            if track not in track2int:
                continue
            x[0, 0] = track2int[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            _ = artist_search(
                preds = probabilities[0], 
                candidates = candidates, 
                int2track = int2track, 
                seeds = start_sequence,
                n = n+100)

        
        track_pointer = -1
        track = start_sequence[track_pointer]
        while track not in track2int:
            track_pointer -= 1
            try:
                track = start_sequence[track_pointer]
            except:
                return []

        truth_flag = False
        truth_pointer = 0
        valid_sequence = [x for x in start_sequence if x in track2int]

        for n in range(n):
            if self.teacher_forcing == 'full':
                track = np.random.choice(valid_sequence, 1)[0]
            x = np.zeros((1, 1))
            x[0, 0] = track2int[track]
            [probabilities, state] = sess.run(
                [self.softmax, self.state], 
                feed_dict={
                    self.X: x,
                    self.initial_state: state
                })
            track_int = artist_search(
                preds = probabilities[0], 
                candidates = candidates, 
                int2track = int2track, 
                seeds = start_sequence,
                n = n+100)
            
            # Semi-guided prediction
            if truth_flag:
                truth_flag = False
                if truth_pointer == len(valid_sequence):
                    truth_pointer = 0
                track = start_sequence[truth_pointer]
            else:
                truth_flag = True
                track = int2track[track_int]
        

        # return most probable candidates
        return_candidates = [x[0] for x in Counter(candidates).most_common(n)]

        return [x for x in return_candidates if x not in ['<eos>', 'unknown']]


##################################################################
############################## MAIN ##############################
##################################################################


def main():

    # parse arguments for command line training or prediction
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-train', '-t',
        type=bool_parser, 
        nargs='?',
        default='not_set',
        help='train boolean')

    ap.add_argument(
        '-config', '-c', 
        type=str, 
        nargs='?',
        default='config.json',
        help='path to config.json')

    args = vars(ap.parse_args())


    print ('Training flag is set to: {}'.format(args['train']))
    print ('Config file path: {}'.format(args['config']))
    config_arg = args['config']
    training_flag = args['train']

    ##################################################################
    ############################## SETUP #############################
    ##################################################################

    
    t2s_config = load_obj('{}'.format(config_arg), 'json')
    input_folder = t2s_config['RESULTS_FOLDER']  # data of pre-processing steps
    model_folder = t2s_config['T2S_MODEL_FOLDER']  # where model checkpoints are stored
    model_name = t2s_config['T2S_MODEL_NAME']  # name of model
    full_model_path = os.path.join(model_folder, model_name)
    hyper_parameters = t2s_config['HYPER_PARAMETERS']

    # generate folder
    if not os.path.exists(full_model_path):
        print ('Created {} ...'.format(full_model_path))
        os.makedirs(full_model_path)

    print ('Loading data ...')
    data = load_obj(os.path.join(input_folder, 'id_sequence.pckl'), 'pickle')
    vocab = load_obj(os.path.join(input_folder, 'track2id.pckl'), 'pickle')
    track2int = vocab
    int2track = {v:k for k,v in track2int.items()}
    print ('There are {} tokens in the vocabulary'.format(len(int2track)))

    ##################################################################
    ######################### HYPER PARAMETERS #######################
    ##################################################################
    ######## it's best to change these vealues in config.json ########
    ##################################################################

    seq_length = hyper_parameters['seq_length']  # how long are training sequences
    n_batch_size = hyper_parameters['n_batch_size']  # how many sequences per batch
    n_layers = hyper_parameters['n_layers']  # amount of lstm layers
    epochs = hyper_parameters['epochs']  # epochs to train on
    save_steps = hyper_parameters['save_steps']  # after how many steps should the progress be saved
    latent_size = hyper_parameters['latent_size']  # latent size of LSTM and embedding layer
    skips = hyper_parameters['skips']  # how many skips in between sequences 
    learning_rate = hyper_parameters['learning_rate']  # the learning rate for RNN
    training_type = hyper_parameters['training_type']  # full or nce
    teacher_forcing = hyper_parameters['prediction_type']  # fully- or semi-guided teacher forcing

    if training_flag == 'not_set':
        training = hyper_parameters['training']  # is training active - if not, recommendation process starts / continues
    else:
        training = training_flag

    ##################################################################
    ########################## TRAINING SETUP ########################
    ##################################################################

    evaluation_set_fname = os.path.join(
        input_folder, t2s_config['EVAL_SET_FNAME'])
    results_folder = t2s_config['RECOMMENDATION_FOLDER']
    result_fname = os.path.join(
        results_folder, t2s_config['RECOMMENDATIONS_FNAME'])

    if not os.path.exists(results_folder):
        print('Creating results folder: {}'.format(results_folder))
        os.makedirs(results_folder)

    ##################################################################
    ####################### RECOMMENDATION SETUP #####################
    ##################################################################

    challenge_track = t2s_config['TEAM_TRACK']
    team_name = t2s_config['TEAM_NAME']
    contact_info = t2s_config['TEAM_CONTACT']

    ##################################################################
    ############################## MODEL #############################
    ##################################################################


    # in case a specific GPU should be used
    #gpu_options = tf.GPUOptions(visible_device_list='0')
    #config = tf.ConfigProto(gpu_options=gpu_options)
    #sess = tf.Session(config=config)

    sess = tf.Session()
    
    # initialize data generator
    n_vocab = len(int2track)
    bg = SimpleBatchGenerator(
        data=data, 
        seq_length=seq_length, 
        n_batch_size=n_batch_size,
        step=skips,
        store_folder=os.path.join(full_model_path, 'step_point'))
    
    current_epoch = bg.epoch_counter
    step = bg.current_idx

    # intialize model for training
    model = Seq2Track(
        n_batch_size=n_batch_size, 
        seq_length=seq_length, 
        n_vocab=n_vocab, 
        n_layers=n_layers,
        learning_rate=learning_rate,
        latent_size=latent_size,
        training_type=training_type)

    # initialize model for prediction
    # reusing scope for recommendations
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        pred_model = Seq2Track(
            n_batch_size=n_batch_size, 
            seq_length=seq_length, 
            n_vocab=n_vocab, 
            n_layers=n_layers,
            latent_size=latent_size,
            learning_rate=learning_rate,
            teacher_forcing=teacher_forcing,
            recommendation=True)

    # pick up the process where we left off - if possible
    saver = tf.train.Saver(tf.global_variables())
    init_operation = tf.global_variables_initializer()
    sess.run(init_operation)

    # check if a model exists, if so - load it
    if os.path.exists(os.path.join(full_model_path, 'checkpoint')):
        saver.restore(sess, tf.train.latest_checkpoint(full_model_path))

    # training routine
    if training:
        # run epochs
        for e in range(current_epoch, epochs):
            avg_epoch_cost = []  # store average cost per epoch

            # for any epoch initialize state as zeros
            current_state = np.zeros((n_layers, 2, n_batch_size, latent_size))
            for step in range(bg.current_idx, bg.steps_per_epoch):
                X_batch, y_batch = next(bg.generate())  # generate fresh training batch

                if step % 10 == 0:  # show progress every 10 steps
                    start_time = time.time()
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={model.X: X_batch, model.y: y_batch, model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                    end_time = (time.time() - start_time)
                    print ('Epoch: {} - Step: {} / {} - Cost: {} ({:.6f}) - Time: {:.2f}s'.format(
                        e, step, bg.steps_per_epoch, np.mean(avg_epoch_cost), cost, end_time))
                else:
                    cost, _, current_state = sess.run(
                        [model.cost, model.training_op, model.state],
                        feed_dict={
                            model.X: X_batch, 
                            model.y: y_batch, 
                            model.initial_state: current_state})
                    avg_epoch_cost.append(cost)
                
                # Save the model and the vocab
                if step != 0 and step % save_steps == 0:
                    # Save model
                    bg.store_step_counter(step)
                    bg.store_epoch_counter(e)

                    model_file_name = os.path.join(full_model_path, 'model')
                    saver.save(sess, model_file_name, global_step = step)
                    print('Model Saved To: {}'.format(model_file_name))
            # if epoch is over
            bg.store_epoch_counter(e)
            bg.current_idx = 0
            bg.store_step_counter(0)
            model_file_name = os.path.join(full_model_path, 'model')
            saver.save(sess, model_file_name, global_step = step)
            print('Model Saved To: {}'.format(model_file_name))
    
    else:
        pid_collection = extract_pids(result_fname)
        all_challenge_playlists = load_obj(evaluation_set_fname, 'pickle')

        init = tf.global_variables_initializer()
        sess.run(init)
        if os.path.exists(os.path.join(full_model_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(full_model_path))

        num_playlists = 0
        for k in all_challenge_playlists:
            num_playlists += len(all_challenge_playlists[k])

        print('Recommending tracks for {:,} playlists...'.format(num_playlists))

        avg_time = []
        for k in all_challenge_playlists:
            bucket_length = len(all_challenge_playlists[k])
            for ix, playlist in enumerate(all_challenge_playlists[k]):
                start_wall_time = time.time()

                if playlist['pid'] in pid_collection:
                    continue
                reco_per_playlist = []
                reco_store = []

                try:
                    reco_per_playlist = pred_model.recommend(sess, playlist['seed'], int2track, track2int, n=600)
                    if not reco_per_playlist:
                        print('Something went wrong with playlist {}'.format(playlist['pid']))
                        continue
                except KeyboardInterrupt:
                    sys.exit()
                except Exception as err:
                    print('Something went wrong with playlist {} (Error: {})'.format(playlist['pid'], err))
                    continue

                # store recommendations
                reco_per_playlist = reco_per_playlist[:500]
                pid_collection.append(playlist['pid'])
                time_elapsed = time.time() - start_wall_time
                avg_time.append(time_elapsed)

                print(
                    'Recommended {} songs (Bucket {}: {} / {}). Avg time per playlist: {:.2f} seconds.'.format(
                        len(reco_per_playlist),
                        k,
                        ix + 1,
                        bucket_length,
                        np.mean(avg_time)))

                write_recommendations_to_file(challenge_track, team_name, contact_info, playlist['pid'], reco_per_playlist, result_fname)
                
                with open(result_fname, 'a') as f:
                    f.write(str(playlist['pid']) + ', ')
                    f.write(', '.join([x for x in reco_per_playlist]))
                    f.write('\n\n')


if __name__ == "__main__":
    main()
