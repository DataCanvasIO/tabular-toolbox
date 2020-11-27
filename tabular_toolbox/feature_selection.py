# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from collections import defaultdict

from scipy.cluster import hierarchy
from scipy.stats import spearmanr


def select_by_multicollinearity(self, X):
    """
    Adapted from https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
    handling multicollinearity is by performing hierarchical clustering on the features’ Spearman
    rank-order correlations, picking a threshold, and keeping a single feature from each cluster.

    :param self:
    :param X:
    :return:
    """
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)

    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected = [X.columns[v[0]] for v in cluster_id_to_feature_ids.values()]
    unselected = list(set(X.columns.to_list()) - set(selected))

    return corr_linkage, selected, unselected
