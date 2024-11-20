import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm
import json
import pandas as pd

def compute_metrics(
    gallery: np.ndarray,
    targets: np.ndarray,
    K: int,
    dataset : pd.DataFrame =None,
    filters : list=None,
    storing_path: str = None,
    verbose: bool = True,
) -> (float, float, float):
    """
        Computes the Recall, R-precisions and Mean-Average Precision.
        For more information about definitions of these metrics have
        a look to ---> "A Metric Learning Reality Check" paper
        (https://arxiv.org/abs/2003.08505)

    Args:
        gallery: array of embeddings
        targets: array of labels
        K: the number of nearest neighbors to use for each metric.
        storing_path: the file path for printing the metrics values.
    Returns:
        The computed metrics
    """
    tree = KDTree(gallery,metric='euclidean')
    recall = 0.0
    r_precision = 0.0
    mean_avg_prc = 0.0
    targ_old=targets
    K_old=K
    
    if verbose:
        print(f"compute metrics with K equal to {K}...")
    for idx, embedding_query in tqdm(enumerate(gallery)):
        
        targets=targ_old
        K=K_old
        label_query = targets[idx]
        if filters is not None:
            mask=dataset['image']!='0'
            for column in filters:
                value=dataset.loc[idx,column]
                add_mask=dataset[column]==value
                mask= mask & add_mask
            targets=targets[mask]
            gall_new=gallery[mask]
            tree=KDTree(gall_new,metric='euclidean')
        embedding_query = np.expand_dims(embedding_query, axis=0)
        if K<len(targets):
            _, indices_matched = tree.query(embedding_query, k=K + 1,breadth_first=True)
        else:
            K=len(targets)-1
            _, indices_matched = tree.query(embedding_query, k=K+1)
        indices_matched = indices_matched[0]  # squeeze
        indices_matched_temp = indices_matched[1 : K + 1]
        classes_matched = targets[indices_matched_temp]
        # Recall
        recall += np.count_nonzero(classes_matched == label_query) > 0  #is there at leas one right in the first K
        # R_precision
        R=sum(dataset['SUBJECT_ID']==dataset.loc[idx,'SUBJECT_ID'])
        _, indices_matched2 = tree.query(embedding_query, k=R,breadth_first=True)
        indices_matched2 = indices_matched2[0]  # squeeze
        indices_matched_temp2 = indices_matched2[1 : R] ###mod
        classes_matched2 = targets[indices_matched_temp2]
        r_precision += np.count_nonzero(classes_matched2 == label_query) / (R-1+1e-6) 
        sum_over_precisions = 0 #needed for the summation 
        for i in range(1, R):
            if classes_matched2[i - 1] == label_query:
                precision_at_i = (
                    np.count_nonzero(classes_matched2[:i] == label_query) / i
                )
                sum_over_precisions += precision_at_i
        sum_over_precisions /= (R-1+1e-6)   #changed from K to R-1
        mean_avg_prc += sum_over_precisions
    recall = recall / (1.0 * gallery.shape[0])
    r_precision = r_precision / (1.0 * gallery.shape[0])
    mean_avg_prc = mean_avg_prc / (1.0 * gallery.shape[0])

    recall *= 100.0
    r_precision *= 100.0
    mean_avg_prc *= 100.0
    
    if storing_path is not None:
        with open(storing_path, "w") as f:
            metrics = {
                f"Recall@{K}": f"{recall:.2f}%",
                f"{K}-precision": f"{r_precision:.2f}%",
                f"MAP@{K}": f"{mean_avg_prc:.2f}%",
            }
            json.dump(metrics, f, indent=4)
        f.close()
    return recall, r_precision, mean_avg_prc
