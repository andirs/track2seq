import nltk
import numpy as np
import os
import pandas as pd

from collections import Counter
from tools.io import load_obj, store_obj
from tools.levenshtein import Levenshtein

print ('')
print ('#' * 80)
print ('Track2Seq Levenshtein Seeds')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

t2s_config = load_obj('config.json', 'json')  # all configuration files can be set manually as well
RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']  # all information will be stored here
RANDOM_STATE = t2s_config['RANDOM_STATE']
recompute = True  

np.random.seed(RANDOM_STATE)

##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":
    playlist_df_fname = os.path.join(RESULTS_FOLDER, 'playlist_df.csv')
    x_train_pids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl')
    dev_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'dev_playlist_dict.pckl')
    test_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'test_playlist_dict.pckl')
    all_playlists_dict_fname = os.path.join(RESULTS_FOLDER, 'all_playlists_dict.pckl')

    playlist_df = pd.read_csv(playlist_df_fname, index_col=0)
    x_train_pids = load_obj(x_train_pids_fname, 'pickle')
    
    # LEVENSHTEIN
    print ('Computing Levenshtein distance ...')
    train_playlist_df = playlist_df[playlist_df.pid.isin(x_train_pids)].copy()
    train_playlist_df.reset_index(inplace=True)
    del(playlist_df)

    print ('Preprocessing all playlist names in training set ...')
    all_playlist_names = train_playlist_df['name'].apply(Levenshtein.pre_process)

    print ('Loading evaluation data set ...')
    dev_playlist_dict = load_obj(dev_playlist_dict_fname, 'pickle')
    test_playlist_dict = load_obj(test_playlist_dict_fname, 'pickle')
    zero_test = test_playlist_dict[0]
    
    print ('Starting with Levenshtein seeds ...')
    print ('Loading all playlists ...')
    all_playlists_dict = load_obj(all_playlists_dict_fname, 'pickle')
    # iterate over first 0-seed playlists
    
    print ('Working on test seeds ...')
    test_leve_seed_dict = Levenshtein.generate_levenshtein_seed_dict(
        zero_test, all_playlist_names, all_playlists_dict, 
        train_playlist_df, RESULTS_FOLDER, 'test_leve_seed_dict.pckl', recompute, seed_k=100)  # change to 500 to get recommendation format
    print ('')

print ('#' * 80)
print ('Finished Track2Seq Levenshtein Seeds')
print ('#' * 80)
print ('')
