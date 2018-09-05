"""
Metrics module for RecSys 2018 challenge. Implements r-precision,
normalized discounted cumulative gain, recommend songs count and recall.
"""

import numpy as np
from collections import OrderedDict
from collections import namedtuple

def r_precision(ground_truth, prediction):
    """
    R-precision is the number of retrieved relevant tracks divided by the number of known relevant tracks.

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations. Usually a list of elements that have been hidden from a particular playlist.
    prediction: list of predictions a given algorithm returns.
    
    Returns:
    ------------
    relevant_tracks: list of all relevant tracks in order of appearance in prediction set
    r-precision metric: float measure of r-precision
    """
    relevant_tracks = []
    for idx, track in enumerate(prediction):
        if track in ground_truth and idx < len(ground_truth):
            relevant_tracks.append(track)
    return relevant_tracks, (len(relevant_tracks) / float(len(ground_truth)))


def get_relevance(ground_truth, item):
    """
    Returns relevance measure for playlist predictions.

    Parameters:
    ------------
    ground_truth: list of elements representing relevant recommendations.
    item: recommendation that needs to be checked

    Returns:
    ------------
    relevance: 1 if track is in ground_truth, 0 otherwise
    """
    if item in ground_truth:
        return 1
    return 0


# dcg and ndcg as per RcsysChallengeTool
# https://github.com/plamere/RecsysChallengeTools/blob/master/metrics.py
def dcg(relevant_elements, retrieved_elements, k=500, *args, **kwargs):
    """Compute the Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Returns:
        DCG value
    """
    retrieved_elements = __get_unique(retrieved_elements[:k])
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))


def ndcg(relevant_elements, retrieved_elements, k=500, *args, **kwargs):
    r"""Compute the Normalized Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.
    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Returns:
        NDCG value
    """

    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    idcg = dcg(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg


def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """
    return list(OrderedDict.fromkeys(original_list))


# playlist extender clicks
def rsc(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    i = set(predictions).intersection(set(targets))
    for index, t in enumerate(predictions):
        for track in i:
            if t == track:
                return float(int(index / 10))
    return float(max_n_predictions / 10.0 + 1)


def recall(ground_truth, prediction):
    """
    Returns recall for a given retrieval task. 
    Recall can be defined as the number of relevant predictions given
    all relevant documents. 

    Parameters:
    --------------
    ground_truth: list, elements representing all known relevant items
    prediction:   list, predictions
    """
    return len(set(prediction).intersection(set(ground_truth))) / len(ground_truth)


def evaluate(pred_set, test_set, exclude_cold=False):
    """
    RecSys specific evaluation method. Returns a dictionary
    with a summary of all metric calculations.

    Parameters:
    --------------
    pred_set:     dict, {'k': []} k = seed bucket, maps to list of lists with 500 recommendations each
    test_set:     dict, {'k': []}
    exclude_cold: bool, flag if set True, 0 seed is being excluded

    Returns:
    --------------
    result_dict:  dict, {'metric_name': float}
    """
    result_dict = {}
    for key in test_set.keys():
        if exclude_cold and key == 0 or key not in pred_set:
            continue
        result_dict[key] = {}
        all_r_precs = []
        all_ndcg = []
        all_rsc = []
        all_recall = []
        preds = pred_set[key]
        gt = [x['groundtruth'] for x in test_set[key]]
        if len(gt) == 0 or len(preds) == 0:
            continue
        for x, y in zip(gt, preds):
            if len(x) == 0 or len(y) == 0:
                continue
            all_r_precs.append(r_precision(x, y)[1])
            all_ndcg.append(ndcg(x, y))
            all_rsc.append(rsc(x, y))
            all_recall.append(recall(x, y))
        result_dict[key]['r_precision'] = np.mean(all_r_precs)
        result_dict[key]['ndcg'] = np.mean(all_ndcg)
        result_dict[key]['rsc'] = np.mean(all_rsc)
        result_dict[key]['recall'] = np.mean(all_recall)
    return result_dict


def print_results(result_dict):
    """
    Prints recommendation result statement.

    Parameters:
    --------------
    result_dict: dict, output from evaluate method {'metric': float}

    Returns:
    --------------
    None
    """
    print ('{:<20}{:<20}{:<20}{:<20}{:<20}'.format('k', 'r_precision', 'ndcg', 'rsc', 'recall'))
    print ('='*100)
    sorted_keys = sorted([int(x) for x in result_dict.keys()])
    for k in sorted_keys:
        print ('{:<20}{:<20.4f}{:<20.4f}{:<20.4f}{:<20.4f}'.format(
            k, result_dict[k]['r_precision'], 
            result_dict[k]['ndcg'], 
            result_dict[k]['rsc'],
            result_dict[k]['recall']))


def main():
    # calc r-precision
    ground_truth = ['1', '2', '3', '5', '8', '99']
    prediction = ['5', '8', '13', '3']
    print (r_precision(ground_truth, prediction))

    # calc normalized discounted cumulative gain
    print(ndcg(ground_truth, prediction))

    # calculate recommended songs count
    ground_truth_rsc_one = [1]
    ground_truth_rsc_two = [499]
    ground_truth_rsc_three = [500]
    prediction_rsc_one = range(500)
    
    print (rsc(ground_truth_rsc_one, prediction_rsc_one))

if __name__ == "__main__":
    main()
