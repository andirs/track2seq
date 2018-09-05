import os
import json
from tools.io import load_obj

def open_json(file_name):
    """
    Helper method to open and return json
    file.
    
    Parameters:
    --------------
    file_name: string representation of path to json file
    
    Returns:
    --------------
    tmp_playlist_json: json object
    """
    with open(file_name, 'r') as f:
        tmp_playlist_json = json.load(f)
    return tmp_playlist_json


class RecSysHelper:
    """
    Helper class to retrieve playlist information
    from complete database.
    """
    def __init__(
        self, 
        playlist_folder, 
        uri_dict_fname='../../analysis/results/uri_to_artist_and_songname_full.pckl', 
        artist_uri_dict_fname='../../analysis/results/artist_uri_to_string.pckl'):
        self.playlist_folder = playlist_folder
        self.uri_dict = None
        self.artist_uri_dict = None
        self.uri_dict_fname = uri_dict_fname
        self.artist_uri_dict_fname = artist_uri_dict_fname
    

    def get_playlist_filename_and_idx(self, pid):
        """
        Constructs playlist filename and playlist index within file from playlist id
        """
        idx = pid % 1000
        filename_root = int(pid / float(1000)) * 1000
        filename = 'mpd.slice.{}-{}.json'.format(filename_root, filename_root+999)
        return (filename, idx)
    

    def get_playlist(self, pid, tracks_only=False):
        """
        Retrieve playlist with pid from database.
        """
        if pid < 0 or pid > 999999:
            raise ValueError('Playlist id out of range (has to be within 0 and 999999)')
        file_name, playlist_idx = self.get_playlist_filename_and_idx(pid)
        playlist_json = open_json(os.path.join(self.playlist_folder, file_name))
        playlist_collection = playlist_json['playlists']
        if tracks_only:
            return playlist_collection[playlist_idx]['tracks']
        return playlist_collection[playlist_idx]


    def track_uri_to_artist_and_title(self, uri):
        if not self.uri_dict:
            print ('Loading URI dict...')
            self.uri_dict = load_obj(self.uri_dict_fname, 'pickle')
        return self.uri_dict[uri]


    def artist_uri_to_artist_string(self, uri):
        if not self.artist_uri_dict:
            print ('Loading Artist URI dict...')
            self.artist_uri_dict = load_obj(self.artist_uri_dict_fname, 'pickle')
        return self.artist_uri_dict[uri]

