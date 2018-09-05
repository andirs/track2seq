import gensim
import numpy as np
import os
import string
import pandas as pd

from collections import Counter
from tools.io import load_obj, store_obj, extract_pids, write_to_file

print ('#' * 80)
print ('Track2Seq CWVA Seeds')
print ('#' * 80)

##################################################################
############################## SETUP #############################
##################################################################

t2s_config = load_obj('config.json', 'json')  # all configuration files can be set manually as well
PLAYLIST_FOLDER = t2s_config['PLAYLIST_FOLDER']  # set folder of playlist information
RESULTS_FOLDER = t2s_config['RESULTS_FOLDER']  # all information will be stored here
W2V_FOLDER = t2s_config['W2V_FNAME']
RANDOM_STATE = t2s_config['RANDOM_STATE']
recompute = True  

np.random.seed(RANDOM_STATE)

# download `GoogleNews-vectors-negative300.bin.gz` from 
# https://github.com/mmihaltz/word2vec-GoogleNews-vectors
w2v_fname = t2s_config['W2V_BINARY_FNAME']


    # load decoding dicts
emoji_dict = load_obj('dicts/emoji_dict.pckl', 'pickle')
urban_dict = load_obj('dicts/urban_dict.pckl', 'pickle')


##################################################################
############################# METHODS ############################
##################################################################


def remove_punct(token):
    """
    Pre-processing method that removes punctuation characters.
    
    Parameters:
    --------------
    token:          str, token of playlist title 
    
    Returns:
    --------------
    new_string:     str, token with removed punctuation chars
    """
    new_string = ''
    for c in token:
        if c in set(string.punctuation).difference(set('_')):
            continue
        elif c in ['_']:
            new_string += ' '
        else:
            new_string += c
    return new_string.strip()


def check_whitespace_word(input_string):
    """
    Compresses playlists that are white-space expanded such as:
    'P L A Y L I S T' into 'PLAYLIST'.
    
    Parameters:
    --------------
    input_string:   str, playlist title
    
    Returns:
    -------------- 
    input_string:   str, compressed string if title is expanded version
    """
    ws_counter = {'space': 0,
                  'other': []}
    for c in input_string:
        if c == ' ':
            ws_counter['space'] += 1
        else:
            ws_counter['other'].append(c)
    if not len(ws_counter['other']) - 1 > ws_counter['space']:
        return ''.join(ws_counter['other'])
    return input_string


def pre_process_title(title):
    """
    Pre-processing method for playlist title.
    
    Parameters:
    --------------
    title:          str, playlist title
    
    Returns:
    -------------- 
    return_tokens:  list, cleaned playlist tokens
    """
    t = str(title).lower().strip()
    for word in urban_dict:
        if word in t:
            t = t.replace(word, ' '.join(urban_dict[word]))
    for emoji in emoji_dict:
        if emoji in t:
            t = t.replace(emoji, ' ' + ' '.join(emoji_dict[emoji]))
    t = check_whitespace_word(t)
    t = t.split(' ')
    t = [x.strip() for x in t]
    stop_words = ['playlist', 'music']
    t = [remove_punct(x) for x in t]
    return_tokens = []
    for ti in t:
        if ti in urban_dict:
            return_tokens.extend(urban_dict[ti])
        if ti not in stop_words:
            return_tokens.append(ti)
    return return_tokens


def get_vecs(row, return_vecs):
    """
    Generates a pid vector dictionary by iterating over DataFrame.
     
    Parameters:
    --------------
    row:          pd.Series, row of playlist dataframe
    return_vecs:  dict, dictionary that maps from pid to return vector
    
    Returns:
    -------------- 
    None
    """
    playlist_title = row['name']
    playlist_tokens = pre_process_title(playlist_title)
    return_vec = []
    for token in playlist_tokens:
        try:
            return_vec.append(np.array(model.wv.word_vec(token)))
        except KeyError as e:
            continue
    return_vecs[row['pid']] = return_vec


def mean_and_unify(vector_dict):
    """
    Calculates mean of vector representations of several tokens.

    Parameters:
    --------------
    vector_dict:        dict, pid to vector list mappings

    Returns:
    -------------- 
    new_vector_dict:    dict, pid to vector average
    """
    new_vector_dict = {}
    for i in vector_dict:
        length = len(np.array(vector_dict[i]))
        if length > 1 and length < 300:
            new_vector_dict[i] = np.mean(vector_dict[i], axis=0)
        else:
            new_vector_dict[i] = np.array(vector_dict[i]).flatten()
    return new_vector_dict


def avg_vector_to_matrix(avg_vectors):
    """
    Generates vector matrix and translation dictionary.

    Parameters:
    --------------
    avg_vectors:        dict, pid to vector averages

    Returns:
    -------------- 
    matrix:             np.darray, matrix that stores vector mappings
    translation_dict:   dict, matrix id to pid mapping
    """
    translation_dict = {}
    matrix = []
    idx = 0
    for item in avg_vectors:
        if len(avg_vectors[item]) > 0:
            translation_dict[idx] = item
            matrix.append(avg_vectors[item])
            idx += 1
    return np.array(matrix), translation_dict


def cos_sim(a, b):
    """
    Simple vectorized cosine similarity implementation.
    
    Parameters:
    --------------
    a:                  np.darray: vector
    b:                  np.darray: vector

    Returns:
    -------------- 
    cosine_similarity:  float, cosine similarity of a and b vectors
    """
    dot_product = np.dot(a, b)
    return dot_product / (np.linalg.norm(a) * np.linalg.norm(b))


def get_vec_by_tokens(tokens):
    """
    Retrieval method for token input. Generates vector based on
    token list. 
    
    Parameters:
    --------------
    tokens:     list, playlist title tokens

    Returns:
    -------------- 
    vector:     np.darray, average vector
    """
    return_vec = []
    for token in tokens:
        try:
            return_vec.append(np.array(model.wv.word_vec(token)))
        except KeyError as e:
            print ('{} not in dataset'.format(token))
            continue
    if len(return_vec) > 1 and len(return_vec) < 300:
        return np.mean(return_vec, axis=0)
    else:
        return np.array(return_vec).flatten()


def get_similar_playlists(title, k=10):
    """
    Retrieve k similar playlists based on title.
    
    Parameters:
    --------------
    title:              str, playlist title

    Returns:
    -------------- 
    recommendations:    list, k playlist pids 
    """
    tokens = pre_process_title(title)
    vec = get_vec_by_tokens(tokens)
    
    dists = []
    for i in range(len(playlist_title_2_vec)):
        dists.append(cos_sim(vec, playlist_title_2_vec[i]))
    arg_sort_dist = np.argsort(dists)[-k:]
    dsts = np.sort(dists)[-k:]
    return [translation_dict[x] for x in arg_sort_dist][::-1], dsts[::-1]


def get_correspondant_list(pid_to_name, seed_k, results_folder, recompute, is_dev=True):
    """
    Retrieves correspondant list for evaluation set. 
    
    Parameters:
    --------------
    pid_to_name:                dict, pid to name mapping
    seed_k:                     int, amount of seed playlists to retrieve
    results_folder:             str, path to results folder
    recompute:                  bool, recompute flag 
    is_dev:                     bool, flag that defines whether to calculate dev or test results

    Returns:
    -------------- 
    correspondant_list:         dict, pid to correspondant playlist mapping list
    correspondant_list_probs:   dict, pid to probability score list
    """

    if is_dev:
        list_fname = os.path.join(results_folder, 'w2v_dev_correspondant_list.pckl')
        probs_fname = os.path.join(results_folder, 'w2v_dev_correspondant_list_probas.pckl')
    else:
        list_fname = os.path.join(results_folder, 'w2v_test_correspondant_list.pckl')
        probs_fname = os.path.join(results_folder, 'w2v_test_correspondant_list_probas.pckl')

    if recompute:
        correspondant_list = {}
        correspondant_list_probs = {}
        for ix, pid in enumerate(pid_to_name):
            
            print ('Retrieving CWVA for \'{}\' ({:.2f} %)'.format(
                pid_to_name[pid], ((ix+1) / len(pid_to_name)) * 100 ), end='\r')
            try:
                playlists, probabilities = get_similar_playlists(pid_to_name[pid], seed_k)
                correspondant_list[pid] = playlists
                correspondant_list_probs[pid] = probabilities
            except KeyboardInterrupt:
                break
            except:
                print ('Something went wrong with playlist: \'{}\' (pid: {})'.format(pid_to_name[pid], pid))
        store_obj(correspondant_list, list_fname, 'pickle')
        store_obj(correspondant_list_probs, probs_fname, 'pickle')
    else:
        correspondant_list = load_obj(list_fname, 'pickle')
        correspondant_list_probs = load_obj(probs_fname, 'pickle')
    
    return correspondant_list, correspondant_list_probs


def get_seed_tracks_probs(old_pid, seed_pid_list, seed_pid_probs, all_playlists_dict, k=100, include_probs=False):
    """
    Retrieve seed tracks based on probability scores.
    
    Parameters:
    --------------
    old_pid:            dict, pid to name mapping
    seed_pid_list:      int, amount of seed playlists to retrieve
    seed_pid_probs:     str, path to results folder
    all_playlists_dict: bool, recompute flag 
    k:                  bool, flag that defines whether to calculate dev or test results
    include_probs:      bool, defines whether output includes probabilty scores

    Returns:
    -------------- 
    candidates:         list, track candidates
    """
    candidate_list = {}

    for pid, prob in zip(seed_pid_list, seed_pid_probs):
        try:
            for track_uri in all_playlists_dict[pid]['tracks']:
                if track_uri == '<eos>':
                    continue
                if track_uri not in candidate_list:
                    candidate_list[track_uri] = prob
                else:
                    candidate_list[track_uri] += prob
        except:
            continue
    if include_probs:
        return [x for x in Counter(candidate_list).most_common(k)]
    return [x[0] for x in Counter(candidate_list).most_common(k)]


##################################################################
############################## MAIN ##############################
##################################################################


if __name__ == "__main__":
    # WORD2VEC - CWVA
    playlist_df_fname = os.path.join(RESULTS_FOLDER, 'playlist_df.csv')
    x_train_pids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl')
    dev_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'dev_playlist_dict.pckl')
    test_playlist_dict_fname = os.path.join(RESULTS_FOLDER, 'test_playlist_dict.pckl')
    all_playlists_dict_fname = os.path.join(RESULTS_FOLDER, 'all_playlists_dict.pckl')
    playlist_title_2_vec_fname = os.path.join(W2V_FOLDER, 'playlist_title_2_vec.pkl')
    translation_dict_fname = os.path.join(W2V_FOLDER, 'translation_dict.pkl')
    
    print ('Loading word2vec embeddings ...')
    if not os.path.exists(w2v_fname):
        raise ValueError(
            'Download pre-computed word embeddings from https://github.com/mmihaltz/word2vec-GoogleNews-vectors')
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_fname, binary=True)

    if False:
        x_train_pids = load_obj(x_train_pids_fname, 'pickle')
        print ('Calculating average tokens for playlist titles ...')
        playlist_df = pd.read_csv(playlist_df_fname, index_col=0)
        df = playlist_df[playlist_df['pid'].isin(x_train_pids)]
        del(playlist_df)
    
        return_vecs = {}
        _ = df.apply(get_vecs, axis=1, args=(return_vecs, ))
        return_vecs_norm = mean_and_unify(return_vecs)

        playlist_title_2_vec, translation_dict = avg_vector_to_matrix(return_vecs_norm)

        store_obj(playlist_title_2_vec, playlist_title_2_vec_fname, 'pickle')
        store_obj(translation_dict, translation_dict_fname, 'pickle')
    else:
        playlist_title_2_vec = load_obj(playlist_title_2_vec_fname, 'pickle')
        translation_dict = load_obj(translation_dict_fname, 'pickle')

    complete_dev_seed_list_fname = os.path.join(W2V_FOLDER, 'complete_dev_seed_list.pckl')
    complete_test_seed_list_fname = os.path.join(W2V_FOLDER, 'complete_test_seed_list.pckl')

    if recompute:
        #print ('Working on development set ...')
        #dev_playlist_dict = load_obj(dev_playlist_dict_fname, 'pickle')
        #zero_dev = dev_playlist_dict[0]
        #dev_pid_to_name = {}
        #for dplaylist in zero_dev:
        #    dev_pid_to_name[dplaylist['pid']] = dplaylist['name']

        #dev_correspondant_list, dev_correspondant_list_probs = get_correspondant_list(
        #    dev_pid_to_name, seed_k=100, results_folder=RESULTS_FOLDER, recompute=recompute)

        print ('Working on test set ...')
        test_playlist_dict = load_obj(test_playlist_dict_fname, 'pickle')
        zero_test = test_playlist_dict[0]
        test_pid_to_name = {}
        for dplaylist in zero_test:
            test_pid_to_name[dplaylist['pid']] = dplaylist['name']

        test_correspondant_list, test_correspondant_list_probs = get_correspondant_list(
            test_pid_to_name, seed_k=100, results_folder=RESULTS_FOLDER, recompute=recompute, is_dev=False)

        del(model)

        # turn playlists into tracks
        all_playlists_dict = load_obj(all_playlists_dict_fname, 'pickle')
        #print ('Completing dev proxis ...')
        #complete_dev_seed_list = {}
        #for p in dev_correspondant_list:
        #    complete_dev_seed_list[p] = get_seed_tracks_probs(
        #        p, dev_correspondant_list[p], dev_correspondant_list_probs[p], all_playlists_dict)

        #dev_leve_seed_dict = load_obj(os.path.join(RESULTS_FOLDER, 'dev_leve_seed_dict.pckl'), 'pickle')

        # if vectors are missing, fill up with levenshtein proxis
        #for pid in [x['pid'] for x in zero_dev]:
        #    if pid not in complete_dev_seed_list:
        #        complete_dev_seed_list[pid] = dev_leve_seed_dict[pid]
        
        #store_obj(complete_dev_seed_list, complete_dev_seed_list_fname, 'pickle')

        print ('Completing test proxis ...')
        complete_test_seed_list = {}
        for p in test_correspondant_list:
            complete_test_seed_list[p] = get_seed_tracks_probs(
                p, test_correspondant_list[p], test_correspondant_list_probs[p], all_playlists_dict)

        test_leve_seed_dict = load_obj(os.path.join(RESULTS_FOLDER, 'test_leve_seed_dict.pckl'), 'pickle')

        # if vectors are missing, fill up with levenshtein proxis
        for pid in [x['pid'] for x in zero_test]:
            if pid not in complete_test_seed_list:
                complete_test_seed_list[pid] = test_leve_seed_dict[pid]
        
        store_obj(complete_test_seed_list, complete_test_seed_list_fname, 'pickle')

    else:
        complete_dev_seed_list = load_obj(complete_dev_seed_list_fname, 'pickle')
        complete_test_seed_list = load_obj(complete_test_seed_list_fname, 'pickle')
    if recompute:
        # add seeds to dev playlists
        #for p in dev_playlist_dict[0]:
        #    if p['pid'] in complete_dev_seed_list:
        #        p['seed'] = complete_dev_seed_list[p['pid']]

        for p in test_playlist_dict[0]:
            if p['pid'] in complete_test_seed_list:
                p['seed'] = complete_test_seed_list[p['pid']]

        print ('Stored final eval set in results folder ...')
        #store_obj(dev_playlist_dict, os.path.join(RESULTS_FOLDER, 'filled_dev_playlists_dict.pckl'), 'pickle')
        store_obj(test_playlist_dict, os.path.join(RESULTS_FOLDER, 'filled_test_playlists_dict.pckl'), 'pickle')

    print ('#' * 80)
    print ('Finished CWVA Seeds')
    print ('#' * 80)
    print ('')