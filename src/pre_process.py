import collections
import numpy as np
import os
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from tools.io import load_obj, store_obj
from tools.stats import Statistician
from tools.stats import bucketing_eval_playlists, build_vocabulary, create_stratification_classes, filter_sequence, \
    generate_all_train_playlist_set, load_inclusion_tracks, sequences_to_ids, split_playlist_df

print ('#' * 80)
print ('Track2Seq Preprocessing')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

t2s_config = load_obj('config.json', 'json')  # all configuration files can be set manually as well
PLAYLIST_FOLDER = t2s_config['PLAYLIST_FOLDER']  # set folder of playlist information
RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']  # all information will be stored here
RANDOM_STATE = t2s_config['RANDOM_STATE']
EVAL_SET_SIZE = t2s_config['EVAL_SET_SIZE']
recompute = True  

np.random.seed(RANDOM_STATE)

##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        print ('Results folder created ...')

    # PRE PROCESSING
    statistician = Statistician(PLAYLIST_FOLDER, RESULTS_FOLDER)

    print ('Generating popularity dict ...')
    track_popularity_dict = statistician.create_track_popularity_dict(recompute=recompute)
    # generate sampling information
    # stratification on median track popularity, number of tracks and modified at
    print ('Generating playlist DataFrame including feature aggregations ...')
    playlist_df = statistician.get_playlist_df(recompute=recompute)

    # binning for stratification process
    print ('Generating stratification classes ...')
    playlist_df = create_stratification_classes(playlist_df)

    print ('Splitting data into train, test and dev sets ...')
    x_train_pids, x_dev_pids, x_test_pids = split_playlist_df(
        playlist_df, RANDOM_STATE, statistician.all_playlists_dict, RESULTS_FOLDER, EVAL_SET_SIZE, recompute=recompute)

    print ('Bucketing dev & test playlists ...')
    dev_playlist_dict, test_playlist_dict = bucketing_eval_playlists(
        x_dev_pids, x_test_pids, statistician.all_playlists_dict, RESULTS_FOLDER, recompute=recompute)

    print ('Loading training set ...')
    all_train_playlists = generate_all_train_playlist_set(
        x_train_pids, statistician, RESULTS_FOLDER, recompute=recompute)
    
    c_set_tracks = load_inclusion_tracks(dev_playlist_dict, test_playlist_dict)

    id_sequence_fname = os.path.join(RESULTS_FOLDER, 'id_sequence.pckl')
    track2id_fname = os.path.join(RESULTS_FOLDER, 'track2id.pckl')

    if recompute:
        # load playlists
        x_train = []
        print ('Working on training set ...')
        for p in all_train_playlists:
            tmp_playlist = all_train_playlists[p]['tracks']
            tmp_playlist.append('<eos>')
            x_train.extend(tmp_playlist)

        print ('Extracting sequences and building vocabulary ...')
        track2id, track_sequence = build_vocabulary(x_train)

        print ('Filtering sequences ...')
        track2id, track_sequence = filter_sequence(track_sequence, track2id, 5, c_set_tracks)

        print ('Transforming track-uri sequences in int sequences ...')
        track_sequence = sequences_to_ids(track_sequence, track2id)

        print ('Storing id_sequence file ...')
        store_obj(track_sequence, id_sequence_fname, 'pickle')
        print ('Storing vocabulary file ...')
        store_obj(track2id, track2id_fname, 'pickle')
    else:
        track_sequence = load_obj(id_sequence_fname, 'pickle')
        track2id = load_obj(track2id_fname, 'pickle')

    ## END OF PRE-PROCESSING
    print ('Generated all files for next steps ...')

print ('#' * 80)
print ('Finished Track2Seq Preprocessing')
print ('#' * 80)
print ('')
