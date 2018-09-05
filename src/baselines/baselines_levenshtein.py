#!/usr/bin/env python3
import os
import pandas as pd
import sys
sys.path.append('../')

from tools.io import load_obj, store_obj
from tools.levenshtein import Levenshtein

def main():
    t2s_config = load_obj('../config.json', 'json')
    RESULTS_FOLDER = os.path.join('../', t2s_config['RESULTS_FOLDER'])
    RECOMMENDATION_FOLDER = os.path.join('../', t2s_config['RECOMMENDATION_FOLDER'])
    EVAL_SET_FNAME = t2s_config['EVAL_SET_FNAME']

    playlist_df_fname = os.path.join(RESULTS_FOLDER, 'playlist_df.csv')
    x_train_pids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl')
    
    playlist_df = pd.read_csv(playlist_df_fname, index_col=0)
    x_train_pids = load_obj(x_train_pids_fname, 'pickle')

    train_playlist_df = playlist_df[playlist_df.pid.isin(x_train_pids)].copy()
    train_playlist_df.reset_index(inplace=True)
    del(playlist_df)

    all_playlist_names = train_playlist_df['name'].apply(Levenshtein.pre_process)
    test_playlist_dict = load_obj(os.path.join(RESULTS_FOLDER, EVAL_SET_FNAME), 'pickle')
    all_playlists_dict = load_obj(os.path.join(RESULTS_FOLDER, 'all_playlists_dict.pckl'), 'pickle')

    test_leve_seed_dict = {}
    for k in test_playlist_dict:
        print ('Working on {}'.format(k))
        tmp_dict = Levenshtein.generate_levenshtein_seed_dict(
            test_playlist_dict[k], all_playlist_names, all_playlists_dict, 
            train_playlist_df, RECOMMENDATION_FOLDER, 'levenshtein_recommendations.pckl', True, seed_k=500)
        for p in tmp_dict:
            test_leve_seed_dict[p] = tmp_dict[p]

    store_obj(test_leve_seed_dict, os.path.join(RECOMMENDATION_FOLDER, 'levenshtein_recommendations.pckl'), 'pickle')


if __name__ == "__main__":
    main()