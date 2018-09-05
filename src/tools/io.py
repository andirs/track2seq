import os
import pickle
import json
import pandas as pd

def extract_pids(fname):
    """
    Helper function to read recommendation pids from
    text file.
    """
    pids = []
    if not os.path.exists(fname):
        return pids
    with open(fname, 'r') as f:
        for line in f.readlines()[1:]:
            if line == '\n':
                continue
            if ',' in line:
                pids.append(int(line.split(',')[0]))
    return pids


def bool_parser(v):
    if v == 'not_set':
        return v
    elif v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Please use boolean value (i.e. true/false).')


def write_to_file(pid, items, fname):
    with open(fname, 'a') as f:
        f.write(str(pid) + ', ')
        f.write(', '.join([x for x in items]))
        f.write('\n\n')


def write_recommendations_to_file(challenge_track, team_name, contact_info, pid, recos, fname):
    # check if file already exists 
    if not os.path.exists(fname):
        with open(fname, 'a') as f:
            f.write('team_info,{},{},{}\n'.format(challenge_track, team_name, contact_info))
    write_to_file(pid, recos, fname)


def load_obj(fname, dtype='json'):
    """
    Object loader function to simplify code.
    
    Parameters:
    --------------
    fname: str, file name of stored object
    dtype: str, 'json', 'pickle' or 'pandas'
    
    Returns:
    --------------
    return_obj: depending on dtype returns json object, pickle representation of dict/list or pd.DataFrame
    """
    if not os.path.exists(fname):
        raise IOError('{} does not exist and needs to be recomputed. Set recompute flag to \'True\''.format(fname))
    else:
        if dtype == 'json':
            with open(fname, 'rb') as f:
                return_obj = json.load(f)
        elif dtype == 'pickle':
            with open(fname, 'rb') as f:
                return_obj = pickle.load(f)
        elif dtype == 'pandas':
            return_obj = pd.read_csv(fname, sep='\t', header=None)
        else:
            raise ValueError('Data type {} does not exist. Use json, pickle or pandas'.format(dtype))
        return return_obj


def store_obj(obj, fname, dtype='pickle'):
    """
    Object storing function to simplify code.
    
    Parameters:
    --------------
    fname: str, file name of stored object
    dtype: str, 'json' or 'pickle'
    
    Returns:
    --------------
    None
    """
    folder_fname = os.path.dirname(fname)
    if not os.path.exists(folder_fname):
        print ('{} does not exist and has been created'.format(folder_fname))
        os.makedirs(folder_fname)
    else:
        if dtype == 'pickle':
            with open(fname, 'wb') as f:
                pickle.dump(obj, f)
        elif dtype == 'json':
            with open(fname, 'wb') as f:
                json.dump(obj, f)


def load_challenge_set(fname='../../../workspace/challenge_data/challenge_set.json'):
    return load_obj(fname, 'json')