#!/usr/bin/env python3
import os
import pandas as pd
import sys
sys.path.append('../')

from collections import Counter
from tools.io import load_obj, store_obj

def get_seed_tracks_probs(old_pid, seed_pid_list, seed_pid_probs, all_playlists_dict, k=100, include_probs=False):
    candidate_list = {}
    candidate_counts = 0
    i = 0
    for pid, prob in zip(seed_pid_list, seed_pid_probs):
        try:
            for track_uri in all_playlists_dict[pid]['tracks']:
                if track_uri not in candidate_list:
                    candidate_list[track_uri] = prob
                else:
                    candidate_list[track_uri] += prob
        except:
            continue
    if include_probs:
        return [x for x in Counter(candidate_list).most_common(k)]
    return [x[0] for x in Counter(candidate_list).most_common(k)]

def main():
    t2s_config = load_obj('../config.json', 'json')
    RESULTS_FOLDER = os.path.join('../', t2s_config['RESULTS_FOLDER'])
    RECOMMENDATION_FOLDER = os.path.join('../', t2s_config['RECOMMENDATION_FOLDER'])
    EVAL_SET_FNAME = t2s_config['EVAL_SET_FNAME']

    print ('Loading playlists ...')
    all_playlists_dict = load_obj(os.path.join(RESULTS_FOLDER, 'all_playlists_dict.pckl'), 'pickle')

    w2v_final_correspondants_fname = os.path.join(RESULTS_FOLDER, 'w2v_test_correspondant_list.pckl')
    w2v_final_correspondants_probas_fname = os.path.join(RESULTS_FOLDER, 'w2v_test_correspondant_list_probas.pckl')
    w2v_final_correspondants = load_obj(w2v_final_correspondants_fname, 'pickle')
    w2v_final_correspondants_probas = load_obj(w2v_final_correspondants_probas_fname, 'pickle')
    
    print ('Loading correspondant lists ...')
    final_test_seed_list = {}
    for p in w2v_final_correspondants:
        final_test_seed_list[p] = get_seed_tracks_probs(
            p, w2v_final_correspondants[p], w2v_final_correspondants_probas[p], all_playlists_dict, k=500)

    store_obj(final_test_seed_list, os.path.join(RECOMMENDATION_FOLDER, 'cwva_recommendations.pckl'), 'pickle')


if __name__ == "__main__":
    main()