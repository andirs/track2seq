import gensim
from gensim.models import Word2Vec
import sys
import os
import time
import numpy as np
sys.path.append('../')
from tools.io import extract_pids, load_obj, store_obj, write_recommendations_to_file

def generate_documents(playlists, uri_to_songname_and_artist):
    all_playlists = []
    for playlist in playlists:
        playlist_document = []
        for track in playlist["tracks"]:
            generate_lookup_library(
                uri_to_songname_and_artist, track["track_uri"], track["track_name"], track["artist_name"])
            playlist_document.append(track["track_uri"])
        all_playlists.append(playlist_document)
    return all_playlists, uri_to_songname_and_artist


def get_similar_results(mdl, track_uri, uri_dict, k=10000, max_artist=3, diff_artist=False):
    sim = mdl.wv.most_similar(track_uri, topn=k)
    seed_artist = uri_dict[track_uri][0]
    artist_counter_dict = {}
    recommendations = []
    for el in sim:
        artist_name = uri_dict[el[0]][0]
        if artist_name == seed_artist and diff_artist:
            continue
        if artist_name not in artist_counter_dict:
            artist_counter_dict[artist_name] = 1
        else:
            if artist_counter_dict[artist_name] >= max_artist:
                continue
            artist_counter_dict[artist_name] += 1
        recommendations.append(el[0])
        if len(recommendations) == 500:
            return recommendations
    print ('Less than 500 recommendations.')
    return recommendations

if __name__ == "__main__":
    recompute = True
    uri_to_songname_and_artist_full = {}
    all_docs = []

    store_folder = 'track_w2v/data/'
    model_folder = 'track_w2v/model/'

    t2s_config = load_obj('{}'.format('../config.json'), 'json')
    RECOMMENDATION_FOLDER = os.path.join('../', t2s_config['RECOMMENDATION_FOLDER'])
    RESULTS_FOLDER = os.path.join('../', t2s_config['RESULTS_FOLDER'])

    if not os.path.exists(store_folder):
        os.makedirs(store_folder)

    playlist_path = os.path.join('../', t2s_config['PLAYLIST_FOLDER'])
    x_train_pids_fname = os.path.join(RESULTS_FOLDER, 'x_train_pids.pckl')
    x_train_pids = load_obj(x_train_pids_fname, 'pickle')

    all_playlist_file_fnames = [x for x in os.listdir(playlist_path) if 'mpd' in x]
    n_files = len(all_playlist_file_fnames)

    if recompute:
        all_playlists = []
        uri_to_artist_and_songname_full = {}
        for idx, playlist_file in enumerate(all_playlist_file_fnames):
            print ('Working on {} / {}'.format(idx + 1, n_files), end='                 \r')
            playlists = load_obj(os.path.join(playlist_path, playlist_file), 'json')
            for playlist in playlists['playlists']:
                if playlist['pid'] in x_train_pids:
                    playlist_document = []
                    for track in playlist["tracks"]:
                        playlist_document.append(track['track_uri'])
                        uri_to_artist_and_songname_full[track['track_uri']] = (track['artist_name'], track['track_name'])
                    all_playlists.append(playlist_document)
        
        print ('Storing playlist sentence file ...')
        store_obj(all_playlists, os.path.join(store_folder, 'sentences.pckl'))
        store_obj(uri_to_artist_and_songname_full, os.path.join(store_folder, 'uri_to_artist_and_songname_full.pckl'))
    else:
        uri_to_artist_and_songname_full = load_obj(os.path.join(store_folder, 'uri_to_artist_and_songname_full.pckl'), 'pickle')
        all_playlists = load_obj(os.path.join(store_folder, 'sentences.pckl'), 'pickle')

    print ('Generating w2v model ...')
    trackw2v_model = Word2Vec(all_playlists, window=5, sg=1, min_count=1, workers=4)

    print ('Saving w2v model ...')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    trackw2v_model.save(fname_or_handle=os.path.join(model_folder, 'final_w2v_training.mdl'))

    print ('Word2Vec Recommendation Run')
    #print ('Loading Word2Vec Model ...')

    eval_set_fname = os.path.join(RESULTS_FOLDER, 'filled_test_playlists_dict.pckl')
    TRACK_W2V_RESULTS_FOLDER = 'track_w2v/data/'

    if not os.path.exists(TRACK_W2V_RESULTS_FOLDER):
        print ('Creating results folder: {}'.format(TRACK_W2V_RESULTS_FOLDER))
        os.makedirs(TRACK_W2V_RESULTS_FOLDER)

    print ('Loading data ...')
    uri_to_artist_and_songname_fname = os.path.join(TRACK_W2V_RESULTS_FOLDER, 'uri_to_artist_and_songname_full.pckl')
    uri_to_artist_and_songname_full = load_obj(uri_to_artist_and_songname_fname, 'pickle')

    eval_set = load_obj(eval_set_fname, 'pickle')

    reco_list = []
    num_buckets = len(eval_set)

    print ('Recommending tracks for {:,} buckets...'.format(num_buckets))
    result_fname = os.path.join(RECOMMENDATION_FOLDER, 'w2v_recommendations.csv')

    if not os.path.exists(result_fname):
        with open(result_fname, 'a') as f:
            f.write('team_info,{},{},{}\n'.format(
                t2s_config['TEAM_TRACK'], 
                t2s_config['TEAM_NAME'], 
                t2s_config['TEAM_CONTACT']))
        pid_collection = []
    else:
        pid_collection = extract_pids(result_fname)

    avg_time = []
    for k in eval_set:
        for ix, playlist in enumerate(eval_set[k]):
            num_playlists = len(eval_set[k])
            start_wall_time = time.time()
            #reco_dict[bucket] = []
            #clear_output(wait=True) 
            if playlist['pid'] in pid_collection:
                continue
            reco_per_playlist = []
            reco_store = []
            for idx, track in enumerate(playlist['seed']):
                try:
                    sim = get_similar_results(trackw2v_model, track, uri_to_artist_and_songname_full, diff_artist=True)
                    reco_store.append(sim)
                except KeyError as e:
                    print ('Track not known')
                    continue
            min_tracks_per_seed = int(500 / len(reco_store))
            
            reco_count = 0
            # for first 1000 playlists, use seed tracks as recommendation as well
            if ix < 1000:
                reco_count = 100
                reco_per_playlist = playlist['seed']
            print ('Working on bucket {} and playlist {:,} of {:,} ({:.2f} %)'.format(
                k, ix, num_playlists, ((ix / num_playlists) * 100)))

            for i in range(len(reco_store[0])):
                for j in range(len(reco_store)):
                    if reco_count == 500:
                        break
                    if reco_store[j][i] not in reco_per_playlist and reco_store[j][i] not in playlist['tracks']:
                        reco_per_playlist.append(reco_store[j][i])
                        reco_count += 1
            if reco_count < 500:
                print ('Too few entries')
            reco_list.append(reco_per_playlist)
            pid_collection.append(playlist['pid'])
            time_elapsed = time.time() - start_wall_time
            avg_time.append(time_elapsed)
            print ('Recommended {} songs. Avg time per playlist: {:.2f} seconds.'.format(len(reco_per_playlist), np.mean(avg_time)))
            with open(result_fname, 'a') as f:
                f.write(str(playlist['pid']) + ', ')
                f.write(', '.join([x for x in reco_per_playlist]))
                f.write('\n\n')