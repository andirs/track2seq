# Track2Seq
Track2Seq is a deep long short term memory network for automated music playlist continuation. Automatic playlist continuation has been coined one of the grand challenges in music recommendation. In general terms, it is the process of automatically adding music tracks to a playlist that fit the characteristics of the original playlist. 

## Requirements
Track2Seq was written in Python 3.6.4 and works with TensorFlow 1.4. A `requirements.txt` file contains all necessary python modules. The most important modules and versions are mentioned below:

* [gensim (3.1.0)](https://radimrehurek.com/gensim/)
* [implicit (0.3.4)](https://github.com/benfred/implicit/)
* [nltk (3.2.5)](https://www.nltk.org/)
* [numpy (1.14.2)](http://www.numpy.org/)
* [pandas (0.21.0)](https://pandas.pydata.org/)
* [scikit-learn (0.19.1)](http://scikit-learn.org)
* [scipy (1.0.1)](https://www.scipy.org/)
* [seaborn (0.8.1)](https://seaborn.pydata.org/)
* [tensorflow-gpu (1.4)](http://tensorflow.org/)

The easiest way to install all requirements is through following command:
```
pip -r requirements.txt
```

## Usage
Track2Seq has an experimental setup. To recompute the results of the thesis "Track2Seq - N-Item Recommendation Using Deep Long Short
Term Memory Networks" please follow these steps:

### Short Version
After installing requirements and setting the path to playlist folder in `config.json` run the following command:

```python src/main.py```

Which in turn runs the following commands in a sequential fashion:

1. ```python src/pre_process.py```
2. ```python src/levenshtein.py```
3. ```python src/cwva.py```
4. ```python src/rnn.py -t true -c config.json```
5. ```python src/rnn.py -t false -c config.json```
6. ```python src/evaluate.py```

In between 5 and 6 feel free to run any baseline scripts in folder `baselines/`.

### 1. Configure input data
To start running the experiment a few things need to be configured. All variables are defined in `config.json`. The experiment is designed to run "out-of-the-box". The only variable that needs to be changed is `PLAYLIST_FOLDER`. `PLAYLIST_FOLDER` should point to the path where the Million Playlist Dataset (MPD) is located.

If you choose to run several experiments (i.e. with different hyper-parameters), creating copies of the config file with a descriptive name (such as `config_25len_1step_nce.json`) and changing the `T2S_MODEL_NAME` to reflect the experiment as well helps keeping everything organized. 

### 2. Pre-process Data
Run `pre_process.py` to extract playlist sequences from data. This will also split the data in training, development and test sets. Development and test sets are transformed to contain seed and ground-truth information.

### 3. Generate seed-substitutes
Some development and test playlists contain no seed tracks. Running `levenshtein.py` creates seed-tracks for those playlists based on Levenshtein distance. By running `cwva.py` seed tracks are generated using CWVA. Rather than performing seed-track approximation at prediction time, those seed tracks are stored in the development and test set json.

### 4. Train Track2Seq
To train Track2Seq the data needs to be pre-processed. Generating seed-substituted is not necessary for this step to work. 

One way of training the network is running following command:
```
python rnn.py -t true -c config.json
```

The `t` and `c` flag don't need to be set. `c` defaults to the standard `config.json` path and `t` defaults to the pre-defined value in the config. 

If the training flag in `config.json` is set to `true` the training command is as easy as `python rnn.py`. 

### 5. Predict Tracks for Test Playlists
After training, tracks can be predicted by either setting the `train` flag in `config.json` to `false` and running `rnn.py` or running `python rnn.py -t false`.

### 6. Compute Baselines
All baselines can be computed by running separate scripts. Scripts are lcoated in `baselines/`. This step is not necessary to train and use Track2Seq. Below are the baselines and their script names:

* `baselines/baselines_levenshtein.py`: Levenshtein similarity for track recommendation. Uses playlist titles only.
* `baselines/baselines_cwva.py`: Cosine-weighted vector average recommendations. Uses playlist titles only.
* `baselines/baselines_w2v.py`: Word2vec inspired recommendation approach.
* `baselines/baselines_wmf.py`: Weighted Matrix Factorization collaborative filtering recommendation.

### 7. Evaluate
After the prediction task is complete, running `evaluate.py` prints the final results for all computed predictions. 

## Further Configurations
Below is a brief explanation of the most important parameters and hyper-parameters that can be changed in `config.json`.

### Parameters
* EVAL_SET_FNAME: Name of evaluation set.
* PLAYLIST_FOLDER: Path to Million Playlist Dataset.
* RANDOM_STATE: Random seed for intialization and choice function.
* RECOMMENDATION_FOLDER: Folder where recommendations are stored. *(c)*
* RECOMMENDATION_FNAME: Name of recommendation .csv file. *(c)*
* RESULTS_FOLDER: Path where results are stored *(c)*
* T2S_MODEL_FOLDER: Path where models are stored *(c)*
* T2S_MODEL_NAME: Name of model and path in T2S_MODEL_FOLDER *(c)*
* TEAM_CONTACT: Contact for challenge submission.
* TEAM_NAME: Team neam for challenge submission.
* TEAM_TRACK: Track selection for challenge (creative or main).
* W2V_BINARY_FNAME: Path to w2v binary.
* W2V_FNAME: Folder where w2v information is stored.

### Hyper-parameters
* seq_length: The length of each training sequence
* n_batch_size: Amount of training sequences per batch
* n_layers: Defines how many layers of LSTM cells will be used
* epochs: Amount of epochs to train for
* training: Boolean flag that determines whether training or recommendation is performed. When set to true, the model starts or continues training
* save_steps: Determines after how many steps the current network is stored to hard-drive. 
* latent_size: Embedding size for internal representation and width of LSTM layers.
* skips: How many tracks are skipped between training sequences.
* learning_rate: Defines learning rate of optimization algorithm
* dropout_keep_prob: Determines how much information is kept through dropout steps. 
* training_type: Determines whether to use complete training or noise-contrastive estimation. Options are `'full'` or `'nce'`.
* prediction_type: Defines the prediction configuration. Options are `'semi'` or `'full'`. 

## Folder Structure
Track2Seq contains a collection of scripts for the experimental setup.
Below is a summary of the most important scripts and files.

* `analysis/`: Thorough analysis regarding data set and sampling methods.
* `src/baselines/`: Contains scripts to calculate baseline results.
* `src/dicts/`: Urban- and emoji-dictionaries for seed approximations.
* `src/tools/`: Supplemental classes and methods that enable storing, similarity calculations, metric calculations, etc.
* `src/w2v/`: Standard folder to store pre-computed word2vec binary in
* `src/config.json`: Main file to setup experiment.
* `src/cwva.py`: Script that calculates CWVA seeds for recommendation task.
* `src/evaluate.py`: Evaluation script.
* `src/levenshtein.py`: Calculates Levenshtein seeds for reommendation task.
* `src/main.py`: Shortcut to run basic experiment without baseline methods.
* `src/pre_process.py`: Pre-processing script to encode and filter playlist sequences.
* `src/rnn.py`: Deep LSTM network structure and training plus recommendation script.
