#!/usr/bin/env python3
import nltk
import numpy as np
import os

from collections import Counter
from tools.io import store_obj, load_obj

class Levenshtein(object):
    """
    Class to calculate playlist title similarity.
    """
    def __init__(self):
        version = '0.1'

    @staticmethod
    def pre_process(playlist_name):
        """
        Preprocess a playlist name through tokenization and 
        lowercase transformation.
        
        Parameters:
        --------------
        playlist_name:  str, playlist title
        
        Return:
        --------------
        playlist_name:  str, combined tokens of playlist title
        """
        try:
            word = [x for x in playlist_name.lower().split()]
            if len(word) > 1:
                word = '_'.join([x for x in word])
            if isinstance(word, list):
                return word[0]
            return word
        except Exception as e:
            return playlist_name


    @staticmethod
    def get_closest(df_row, playlist_name, return_dict, comp_memory):
        """
        Static method that fills up a return 
        dictionary with similarity scores.
        
        Parameters:
        --------------
        df_row:         pd.Series or DataFrame row
        playlist_name:  str, playlist name
        return_dict:    dict, mapping aggregations and summations for playlist name
        comp_memory:    dict, maps distance values for already computed information
        
        Return:
        --------------
        None
        """
        lowest = return_dict['lowest']
        targets = return_dict['targets']
        try:
            if df_row not in comp_memory:
                distance = nltk.edit_distance(df_row, playlist_name)
                comp_memory[df_row] = distance
            else:
                distance = comp_memory[df_row]
        except:
            return_dict['counter'] += 1
            return None
        if not lowest or distance <= lowest[0]:
            lowest.insert(0, distance)
            targets.insert(0, return_dict['counter'])
        return_dict['counter'] += 1


    @staticmethod
    def get_seed_tracks(playlist_df, return_dict, all_playlists_dict, seed_k=100):
        """
        Retrieves seed tracks based on Levenshtein distance candidates.
        
        Parameters:
        --------------
        playlist_df:            pd.DataFrame
        return_dict:            dict, Levenshtein playlist candidates
        all_playlists_dict:     dict, playlist pid to playlist information
        seed_k:                 int, amount of seed tracks to return
        
        Return:
        --------------
        candidate_list:    list, top k seeds 
        """
        sim_count = len([x for x in return_dict['lowest'] if x == return_dict['lowest'][0]])
        tmp_pids = return_dict['targets']

        if sim_count > 100:
            tmp_pids = tmp_pids[:sim_count]
            np.random.shuffle(tmp_pids)
            candidate_list = {}
            for i in range(100):
                tmp_pid = playlist_df[playlist_df.index == tmp_pids[i]]['pid'].values[0]
                tmp_tracks = all_playlists_dict[tmp_pid]['tracks']
                for track_uri in tmp_tracks:
                    if track_uri == '<eos>':
                        continue
                    if track_uri not in candidate_list:
                        candidate_list[track_uri] = 0
                    else:
                        candidate_list[track_uri] += 1
        else:
            candidate_list = {}
            candidate_counts = 0
            i = 0
            while candidate_counts < seed_k and i < len(tmp_pids):

                tmp_pid = playlist_df[playlist_df.index == tmp_pids[i]]['pid'].values[0]
                tmp_tracks = all_playlists_dict[tmp_pid]['tracks']
                for track_uri in tmp_tracks:
                    if track_uri == '<eos>':
                        continue
                    if track_uri not in candidate_list:
                        candidate_list[track_uri] = 0
                        candidate_counts += 1
                    else:
                        candidate_list[track_uri] += 1
                i += 1

        return Counter(candidate_list).most_common(seed_k)


    @staticmethod
    def generate_levenshtein_seed_dict(
        zero_seed_playlists, 
        all_playlist_names, 
        all_playlists_dict, 
        playlist_df, 
        RESULTS_FOLDER, 
        filename, 
        recompute,
        seed_k=100):
        """
        Generates seed dict based on playlist names.
        
        Parameters:
        --------------
        zero_seed_playlists:    list, playlists to similarity on
        all_playlist_names:     pd.Series, all training set playlist names
        all_playlists_dict:     dict, playlist pid to playlist information
        playlist_df:            pd.DataFrame, playlist dataframe
        RESULTS_FOLDER:         str, path to results folder
        filename:               str, name of resulting file
        recompute:              bool, recompute flag
        seed_k:                 int, amount of seed tracks to return
        
        Return:
        --------------
        seed_set:               dict, mapping pid to seed list
        """
        fname = os.path.join(RESULTS_FOLDER, filename)
        if recompute:
            comp_memory = {}
            seed_set = {}
            for idx, playl in enumerate(zero_seed_playlists):
                playlist_name = Levenshtein.pre_process(playl['name'])
                print ('\r{:.2f} % :: Retrieving levenshtein similarities for \'{}\''.format(
                    ((idx + 1) / len(zero_seed_playlists)) * 100, playlist_name), end='')
                return_dict = {}
                return_dict['counter'] = 0
                return_dict['lowest'] = []
                return_dict['targets'] = []
                _ = all_playlist_names.apply(Levenshtein.get_closest, args=(playlist_name, return_dict, comp_memory))
                seeds = Levenshtein.get_seed_tracks(playlist_df, return_dict, all_playlists_dict, seed_k=seed_k)
                seed_set[playl['pid']] = [x[0] for x in seeds]

            store_obj(seed_set, fname, 'pickle')
        else:
            seed_set = load_obj(fname, 'pickle')

        return seed_set
