#!/usr/bin/env python3
import numpy as np
import os

from tools.io import load_obj, store_obj
from tools.metrics import recsys_metrics


##################################################################
############################# METHODS ############################
##################################################################


def evaluate(prediction_set, dev_playlists):
    """
    Evaluates prediction vs. evaluation results. 
    
    Parameters:
    --------------
    prediction_set: dict, mapping pid to track list
    dev_playlists:  dict, mapping pid to eval set dictionary
    
    Returns:
    --------------
    results:        dict, avg metric dictionary for buckets
    significance    dict, metric lists for buckets to calculate significance values between methods
    """
    results = {
        'r_prec': {},
        'ndcg': {},
        'clicks': {},
        'recall': {},
    }
    
    significance = {
        'r_prec': {},
        'ndcg': {},
        'clicks': {},
        'recall': {},
    }
    for k in dev_playlists:
        tmp = []
        ndcg_tmp = []
        clicks_tmp = []
        recall_tmp = []
        for playlist in  dev_playlists[k]:
            if playlist['pid'] in prediction_set:
                preds = prediction_set[playlist['pid']]
                gt = playlist['groundtruth']
                if len(gt) == 0 or len(preds) == 0:
                    continue
                tmp.append(recsys_metrics.r_precision(gt, preds)[1])
                ndcg_tmp.append(recsys_metrics.ndcg(gt, preds, 500))
                clicks_tmp.append(recsys_metrics.rsc(gt, preds, 500))
                recall_tmp.append(recsys_metrics.recall(gt, preds))
        if tmp:
            results['r_prec'][k] = np.mean(tmp)
            results['ndcg'][k] = np.mean(ndcg_tmp)
            results['clicks'][k] = np.mean(clicks_tmp)
            results['recall'][k] = np.mean(recall_tmp)
            significance['r_prec'][k] = tmp
            significance['ndcg'][k] = ndcg_tmp
            significance['clicks'][k] = clicks_tmp
            significance['recall'][k] = recall_tmp
    for metric in results:
        average = np.mean([results[metric][x] for x in results[metric]])
        results[metric]['avg'] = average
    return results, significance


def extract_recos(fname):
    """
    Retrieves recommendations from challenge set format. 
    
    Parameters:
    --------------
    fname:              str, path to recommendation csv
    
    Returns:
    --------------
    correspondant_list: dict, mapping pid to list of tracks 
    """
    correspondant_list = {}
    with open(fname, 'r') as f:
        for line in f.readlines()[1:]:
            if line == '\n':
                continue
            if ',' in line:
                pid = int(line.split(',')[0])
                playlists = [x.strip() for x in line.split(',')[1:]]
                correspondant_list[pid] = playlists
    return correspondant_list


def print_results(res_dict):
    """
    Print results in orderly fashion.
    
    Parameters:
    --------------
    res_dict:   dict, dictionary that maps metric to results 
    
    Returns:
    --------------
    None
    """
    print ('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('k', 'r_precision', 'ndcg', 'clicks', 'recall'))
    print ('='*120)
    sorted_keys = sorted([int(x) for x in res_dict['recall'].keys() if x != 'avg'])
    for a in sorted_keys:
        print ('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            a, res_dict['r_prec'][a], res_dict['ndcg'][a], res_dict['clicks'][a], res_dict['recall'][a]))
    print ('='*120)
    print ('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            'AVG', res_dict['r_prec']['avg'], res_dict['ndcg']['avg'], res_dict['clicks']['avg'], res_dict['recall']['avg']))

def cwva_evaluate(prediction_set, dev_playlists):
    """
    Special method to evaluate CWVA results.
    
    Parameters:
    --------------
    prediction_set: dict, mapping pid to track list
    dev_playlists:  dict, mapping pid to eval set dictionary 
    
    Returns:
    --------------
    results:        dict, metric to average 
    """
    results = {
        'r_prec': [],
        'ndcg': [],
        'clicks': [],
        'recall': [],
    }
    
    for k in dev_playlists:
        tmp = []
        ndcg_tmp = []
        clicks_tmp = []
        recall_tmp = []
        for playlist in  dev_playlists[k]:
            if playlist['pid'] in prediction_set:
                preds = prediction_set[playlist['pid']]
                gt = playlist['groundtruth']
                tmp.append(recsys_metrics.r_precision(gt, preds)[1])
                ndcg_tmp.append(recsys_metrics.ndcg(gt, preds, 500))
                clicks_tmp.append(recsys_metrics.rsc(gt, preds, 500))
                recall_tmp.append(recsys_metrics.recall(gt, preds))
        if tmp:
            results['r_prec'].extend(tmp)
            results['ndcg'].extend(ndcg_tmp)
            results['clicks'].extend(clicks_tmp)
            results['recall'].extend(recall_tmp)
    for metric in results:
        average = np.mean(results[metric])
        results[metric] = average
    return results


def translate_wmf_recos(wmf_reco, eval_playlist, item_id_to_track_uri):
    """
    Standardizes WMF recommendation results. WMF uses a different track encoding 
    due to its matrix factorization method.
    
    Parameters:
    --------------
    wmf_reco:               dict, mapping bucket size to eval list
    eval_playlist:          dict, mapping pid to eval set dictionary
    item_id_to_track_uri:   dict, mapping WMF track id to track uri
    
    Returns:
    --------------
    return_reco_dict:       dict, pid to track list  
    """
    return_reco_dict = {}
    for k in wmf_reco:
        for pid, p in zip(eval_playlist[k], wmf_reco[k]):
            return_reco_dict[pid] = [item_id_to_track_uri[x[0]] for x in p]
    return return_reco_dict


##################################################################
############################## MAIN ##############################
##################################################################


def main():
    print ('')
    print ('#' * 80)
    print ('Starting Evaluation')
    print ('#' * 80)

    key_list = ['r_prec', 'ndcg', 'clicks', 'recall']
    t2s_config = load_obj('config.json', 'json')
    RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']
    EVAL_SET_FNAME = t2s_config['EVAL_SET_FNAME']
    RECOMMENDATION_FOLDER = t2s_config['RECOMMENDATION_FOLDER']
    T2S_RECO_FNAME = t2s_config['RECOMMENDATIONS_FNAME']

    # load evaluation set
    eval_playlist = load_obj(os.path.join(RESULTS_FOLDER, EVAL_SET_FNAME), 'pickle')

    # display levenshtein and cwva results
    cwva_recommendations_fname = os.path.join(RECOMMENDATION_FOLDER, 'cwva_recommendations.pckl')
    output_strings = []
    if os.path.exists(cwva_recommendations_fname):
        cwva_recommendations = load_obj(cwva_recommendations_fname, 'pickle')
        cwva_eval = cwva_evaluate(cwva_recommendations, eval_playlist)
        output_strings.append('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format('CWVA', *[cwva_eval[k] for k in key_list]))
    else:
        print ('CWVA recommendations are not computed. Please run baselines/baselines_cwva.py')
    
    leve_recommendations_fname = os.path.join(RECOMMENDATION_FOLDER, 'levenshtein_recommendations.pckl')
    if os.path.exists(leve_recommendations_fname):
        leve_recommendations = load_obj(leve_recommendations_fname, 'pickle')
        leve_eval, leve_sig = evaluate(leve_recommendations, eval_playlist)
        output_strings.append('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format('Levenshtein', *[leve_eval[k]['avg'] for k in key_list]))
    else:
        print ('Levenshtein recommendations are not computed. Please run baselines/baselines_levenshtein.py')
    
    if output_strings:
        print ('')
        print ('CWVA vs. Levenshtein Distance')
        print ('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('method', 'r_precision', 'ndcg', 'clicks', 'recall'))
        print ('='*120)
        for result_string in output_strings:
            print (result_string)
        print ('')

    # display word2vec results
    w2v_recos_fname = os.path.join(RECOMMENDATION_FOLDER, 'w2v_recommendations.csv')
    if os.path.exists(w2v_recos_fname):
        print ('')
        print ('Word2Vec Recommendation Results')
        w2v_recos = extract_recos(w2v_recos_fname)
        w2v_eval, w2v_sig = evaluate(w2v_recos, eval_playlist)
        print_results(w2v_eval)
        print ('')
    else:
        print ('Word2Vec recommendations are not computed. Please run baselines/baselines_w2v.py')

    # display WMF results
    wmf_recos_fname = os.path.join('baselines', 'prediction_wmf_100_10_18_04_13.pckl')
    if os.path.exists(wmf_recos_fname):
        print ('')
        print ('WMF Recommendation Results')
        wmf_recommendations = load_obj(wmf_recos_fname, 'pickle')
        dev_playlists_wmf = {}
        dev_playlists_wmf_pids = {}
        for k in eval_playlist:
            dev_playlists_wmf[k] = [x['groundtruth'] for x in eval_playlist[k]]
            dev_playlists_wmf_pids[k] = [x['pid'] for x in eval_playlist[k]]
        track_uri_to_item_id = load_obj(os.path.join('baselines', 'track_uri_to_item_id.pckl'), 'pickle')
        item_id_to_track_uri = {v:k for k,v in track_uri_to_item_id.items()}
        wmf_recommendations = translate_wmf_recos(wmf_recommendations, dev_playlists_wmf_pids, item_id_to_track_uri)
        wmf_eval, wmf_sig = evaluate(wmf_recommendations, eval_playlist)
        print_results(wmf_eval)
        print ('')
    else:
        print ('WMF recommendations are not computed. Please run baselines/baselines_wmf.py')


    # display t2s results
    t2s_recos_fname = os.path.join(RECOMMENDATION_FOLDER, T2S_RECO_FNAME)
    if os.path.exists(t2s_recos_fname):
        print ('')
        print ('Track2Seq Recommendation Results')
        t2s_recos = extract_recos(t2s_recos_fname)
        t2s_eval, t2s_sig = evaluate(t2s_recos, eval_playlist)
        print_results(t2s_eval)
        print ('')

if __name__ == '__main__':
    main()
