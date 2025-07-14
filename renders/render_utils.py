import numpy as np
import torch

def rationale(entry):
    facts_str = " and ".join(entry["facts"]) if entry["facts"] else "no salient facts"
    ig_str = ", ".join(entry["top_IG_features"])
    return (f'I performed {entry["action"]} because {facts_str}. '
            f'This occurred at {entry["t"]:.1f}s, during behavior stage {entry["cluster"]}. '
            f'(Top IG: {ig_str})')


def get_cluster_times(explanation_log):
    """
    Return the start and end times for all clusters in the explanation log.
    
    Args:
        explanation_log: A list of information from each timestep

    Returns:
        cluster_times: A dictionary where the clusters are keys and the values are 2-element lists, where the \
        first element is the start time and the second element is the end time.
    """
    cluster_times = {}

    for i, entry in enumerate(explanation_log):
        current_cluster = entry["cluster"]
        if current_cluster not in cluster_times:
            cluster_times[current_cluster] = [entry["t"], entry["t"]]
        else:
            if entry["t"] < cluster_times[current_cluster][0]:
                cluster_times[current_cluster][0] = entry["t"]
            if entry["t"] > cluster_times[current_cluster][1]:
                cluster_times[current_cluster][1] = entry["t"]
    
    return cluster_times

def get_most_frequent_integrated_gradients_in_cluster(explanation_log):
    """
    Return the most frequent integrated gradient (IG) in the first and second slot.
    
    Args:
        explanation_log: A list of information from each timestep

    Returns:
        cluster_top_integrated_gradients_in_clustertimes: A dictionary where the clusters are keys and the values are 2-element lists, \
        where the first element is the most frequent IG in the first slot and the second element is the most frequent IG in the second slot.
    """

    # First, generate two lists per cluster, one for all first place IG, one for all second place IG
    cluster_to_top_ig = {}
    for i, entry in enumerate(explanation_log):
        if not (i == 0 or entry["cluster"] != explanation_log[i-1]["cluster"]):
            if entry["cluster"] not in cluster_to_top_ig:
                cluster_to_top_ig[entry["cluster"]] = {"first" : [], "second" : []}
            cluster_to_top_ig[entry["cluster"]]["first"].append(entry["top_IG_features"][0])
            cluster_to_top_ig[entry["cluster"]]["second"].append(entry["top_IG_features"][1])
    
    # Second, for each cluster, get the most frequent IG in each list
    top_integrated_gradients_in_cluster = {}
    for cluster, features in cluster_to_top_ig.items():
        cluster_first_ig = np.array(features["first"])
        cluster_second_ig = np.array(features["second"])

        # Get the number of steps for each integrated gradient
        unique_first, counts_first = np.unique(cluster_first_ig, return_counts=True)
        unique_second, counts_second = np.unique(cluster_second_ig, return_counts=True)

        # Output the most frequent first/second integrated gradient
        top_first = unique_first[np.argmax(counts_first)]
        top_second = unique_second[np.argmax(counts_second)]

        top_integrated_gradients_in_cluster[cluster] = {"first" : top_first, "second" : top_second}

    return top_integrated_gradients_in_cluster


# Convert seconds to minute:second strings
def sec_to_minsec(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

def steps_to_minsec(steps, step_size):
    sec = steps * step_size
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"