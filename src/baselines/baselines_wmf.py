import implicit
import numpy as np
import pandas as pd
import pickle
import json
import os
import scipy
import sys
sys.path.append('../')

from scipy.sparse import csr_matrix, lil_matrix
from tools.io import load_obj, store_obj
from tools.metrics import recsys_metrics


##################################################################
############################## SETUP #############################
##################################################################


t2s_config = load_obj('../config.json', 'json')

recompute = True
baseline_results_folder = 'wmf/'
RESULTS_FOLDER = os.path.join('../', t2s_config['RESULTS_FOLDER'])
EVAL_SET_FNAME = t2s_config['EVAL_SET_FNAME']
test_playlist_fname = os.path.join(RESULTS_FOLDER, 'test_playlist_dict.pckl')
RECOMMENDATION_FOLDER = os.path.join('../', t2s_config['RECOMMENDATION_FOLDER'])

if not os.path.exists(baseline_results_folder):
    os.makedirs(baseline_results_folder)


##################################################################
######################### HYPER PARAMETERS #######################
##################################################################

# define hyper-parameter for alternating least-squares model
als_model_dict = {
    'two': {
        'factors': 100,
        'regularization': 10.,
        'use_gpu': True,
        'calculate_training_loss': True,
        'model_fname': 'model_wmf_100_10_18_04_13.pckl',
        'prediction_fname': 'prediction_wmf_100_10_18_04_13.pckl'}
    }


##################################################################
############################# METHODS ############################
##################################################################


def prepare_data_full_cf(
    df_filename = os.path.join(baseline_results_folder, 'playlist_train.csv'),
    playlist_src_folder = os.path.join('../', t2s_config['PLAYLIST_FOLDER']),
    item_dict_filename = os.path.join(baseline_results_folder, 'track_uri_to_item_id.pckl'),
    user_dict_filename = os.path.join(baseline_results_folder, 'playlist_id_to_pidx.pckl'),
    test_playlist_fname = os.path.join(RESULTS_FOLDER, 'filled_dev_playlists_dict.pckl'),
    train_pid_ids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl'),
    test_pidx_row_dict_fname = os.path.join(baseline_results_folder, 'test_pidx_row_dict.pckl'),
    recompute=True):
    """
    Prepares a list of lists where every individual list stores track ids. 
    Also stores pid to match information at a later point.
    
    Parameters:
    --------------
    recompute: bool flag which determines if stored information should be used
    
    Returns:
    --------------
    res_df:       pd.DataFrame, mapping user to item interaction
    item_dict:    dict, item id to track uri
    user_dict:    dict, simplified playlist id to pid
    """

    if recompute:
        counter = 0
        total_files = len(os.listdir(playlist_src_folder))
        
        list_of_list = []
        item_dict = {}
        user_dict = {}
        item_counter = 0

        playlists_of_tracks_uri = []
        pidx = 0
        
        train_pid_ids_dict = load_obj(train_pid_ids_fname, dtype='pickle')
        
        for playlist_json in os.listdir(playlist_src_folder):
            print ("Working on slice {} ({:.2f} %) (File Name:  {} || Total Slices: {})".format(
                    counter, (counter / total_files) * 100, playlist_json, total_files), end='\r')
            
            counter += 1
            data_json = load_obj(os.path.join(playlist_src_folder, playlist_json), dtype='json')

            for playlist in data_json['playlists']:
                if playlist['pid'] not in train_pid_ids_dict:
                    continue  # filter out any test and dev playlists
                
                if playlist['pid'] not in user_dict:
                    user_dict[playlist['pid']] = pidx
                    pidx += 1
                
                for track in playlist['tracks']:
                    if track['track_uri'] in item_dict:
                        track_id = item_dict[track['track_uri']]
                    else:
                        track_id = item_counter
                        item_dict[track['track_uri']] = track_id
                        item_counter += 1
                    list_of_list.append([user_dict[playlist['pid']], track_id, 1])  # pid, track_id, rating
                    
        # add dev set to matrix and dicts
        print ('Loading Test/Dev Set...')
        test_pidx_row_dict = {}
        test_set = load_obj(test_playlist_fname, 'pickle')
        
        for key in [0, 1, 5, 10, 25, 100]:
            list_of_dev_playlists = test_set[key]
            test_pidx_row_dict[key] = []
            
            for playlist in list_of_dev_playlists:
                if len(playlist['seed']) < 1:
                    continue  # filter out any 0 seed playlists
                if playlist['pid'] not in user_dict:
                    test_pidx_row_dict[key].append(pidx)
                    user_dict[playlist['pid']] = pidx
                    pidx += 1

                for track in playlist['seed']:
                    if track in item_dict:
                        track_id = item_dict[track]
                    else:
                        track_id = item_counter
                        item_dict[track] = track_id
                        item_counter += 1
                    list_of_list.append([user_dict[playlist['pid']], track_id, 1])  # pid, track_id, rating
        
        print ('Storing WMF dictionaries ...')
        # store results
        with open(item_dict_filename, 'wb') as f:
            pickle.dump(item_dict, f)
        with open(user_dict_filename, 'wb') as f:
            pickle.dump(user_dict, f)
        with open(test_pidx_row_dict_fname, 'wb') as f:
            pickle.dump(test_pidx_row_dict, f)

        res_df = pd.DataFrame(list_of_list)
        res_df.to_csv(df_filename, sep='\t', index=False, header=False)
    else:
        # load results
        res_df = load_obj(df_filename, dtype='pandas')
        item_dict = load_obj(item_dict_filename, dtype='pickle')
        user_dict = load_obj(user_dict_filename, dtype='pickle')
        test_pidx_row_dict = load_obj(test_pidx_row_dict_fname, dtype='pickle')
    return res_df, item_dict, {v:k for k, v in user_dict.items()}, test_pidx_row_dict


def generate_matrix(
    df, 
    sparse_df_fname=os.path.join(baseline_results_folder, 'df_sparse_matrix.npz'), 
    recompute=True):
    """
    Creates sparse matrix based on interaction DataFrame.
    
    Parameters:
    --------------
    df:          pd.DataFrame, first column after index: user_id; second colum: item_id; third: rating
    recompute:   bool, flag for recomputation
    
    Returns:
    --------------
    df_matrix:   sparse matrix through linked list implementation
    """
    if recompute:
        n_playlists = len(df[0].unique())
        n_tracks = len(df[1].unique())
        df_matrix = lil_matrix((n_playlists, n_tracks))
        df_len = len(df)
        perc = int(df_len / 100)
        for counter, row in enumerate(df.itertuples()):
            if counter % perc == 0:
                print ('{} % '.format(counter / perc), end='\r')
            df_matrix[row[1], row[2]] = 1

        print ('Writing file to hdd...')
        df_matrix = df_matrix.transpose()  # this could be implemented directly in generate matrix
        df_csr = csr_matrix(df_matrix.tocsr())
        with open(sparse_df_fname, 'wb') as f:
            scipy.sparse.save_npz(f, df_csr, compressed=True)
        return df_csr
    else:
        with open(sparse_df_fname, 'rb') as f:
            df_csr = scipy.sparse.load_npz(f)
    return df_csr


def train_and_predict(df_matrix, dev_set, dev_pidx_row_dict, model_dict, recompute=False, exclude_cold=False):
    
    prediction_fname = model_dict['prediction_fname']
    model_fname = model_dict['model_fname']
    
    # define estimator
    als = implicit.als.AlternatingLeastSquares(
        factors=model_dict['factors'], 
        regularization=model_dict['regularization'], 
        use_gpu=model_dict['use_gpu'], 
        calculate_training_loss=model_dict['calculate_training_loss'])
    
    if recompute:
        print ('Fitting model ...')
        als.fit(df_matrix)
        prediction_results = {}
        for key in dev_set.keys():
            if exclude_cold and key == 0:
                continue
            prediction_results[key] = [] 
            df_len = len(dev_pidx_row_dict[key])
            perc = int(df_len / 100)
            for counter, playlist_row_id in enumerate(dev_pidx_row_dict[key]):
                if perc != 0 and counter % perc == 0:
                    print ('Predicting: {} % (k = {})'.format(counter / perc, key), end='\r')
                preds = als.recommend(playlist_row_id, df_matrix, N=500)
                prediction_results[key].append(preds)
        with open(os.path.join(baseline_results_folder, prediction_fname), 'wb') as f:
            pickle.dump(prediction_results, f)
        with open(os.path.join(baseline_results_folder, model_fname), 'wb') as f:
            pickle.dump(als, f)
    else:
        prediction_results = load_obj(os.path.join(baseline_results_folder, prediction_fname), 'pickle')
        als = load_obj(os.path.join(baseline_results_folder, model_fname), 'pickle')
    
    return prediction_results, als


def print_results(result_dict):
    print ('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('k', 'r_precision', 'ndcg', 'rsc', 'recall'))
    print ('='*100)
    sorted_keys = sorted([int(x) for x in result_dict.keys()])
    for k in sorted_keys:
        print ('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            k, result_dict[k]['r_precision'], 
            result_dict[k]['ndcg'], 
            result_dict[k]['rsc'],
            result_dict[k]['recall']))


if __name__ == "__main__":

    results_collection = {}
    results_collection['test'] = {}
    print ('Predicting test set ...')
    df, track_uri_to_item_id, pidx_to_pid, test_pidx_row_dict = prepare_data_full_cf(
        test_playlist_fname = os.path.join(RESULTS_FOLDER, 'filled_test_playlists_dict.pckl'),
        recompute=recompute)
    item_id_to_track_uri = {v:k for k, v in track_uri_to_item_id.items()}
    
    df_matrix = generate_matrix(df, recompute=recompute) 

    # predict
    test_set = load_obj(test_playlist_fname, 'pickle')
    prediction_results_two, als_two = train_and_predict(
        df_matrix, test_set, test_pidx_row_dict, als_model_dict['two'], recompute=recompute)

    pred_set_two = {}

    for key in prediction_results_two:
        list_with_each_500_predictions = prediction_results_two[key]
        pred_set_two[key] = []
        for playlist in list_with_each_500_predictions:
            pred_set_two[key].append([item_id_to_track_uri[x[0]] for x in playlist])

    result_dict_two = recsys_metrics.evaluate(pred_set_two, test_set, exclude_cold=False) 

    results_collection['test']['WMF_2'] = dict(result_dict_two)

    # store results
    store_obj(results_collection, os.path.join(RECOMMENDATION_FOLDER, 'wmf_reco_results.pckl'), 'pickle')

    # print results
    for key in results_collection:
        print ('Data Set: {}'.format(key))
        for k in results_collection[key]:
            print ('Model Name: {}'.format(k))
            print_results(results_collection[key][k])
            print ()
