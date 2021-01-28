
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.sparse import lil_matrix
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler

from .experiment_setup_lib import log_it

class DataStorage:
    def __init__(self, df=None, **kwargs):
        self.__dict__.update(kwargs)

        if self.RANDOM_SEED != -1:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

               len_train_labeled = len(self.labeled_mask)
        len_train_unlabeled = len(self.unlabeled_mask)
        #  len_test = len(self.X_test)

        len_total = len_train_unlabeled + len_train_labeled  # + len_test

        log_it(
            "size of train  labeled set: %i = %1.2f"
            % (len_train_labeled, len_train_labeled / len_total)
        )
        log_it(
            "size of train unlabeled set: %i = %1.2f"
            % (len_train_unlabeled, len_train_unlabeled / len_total)
        )

        log_it("Loaded " + str(self.DATASET_NAME))
   
    def _label_samples_without_clusters(self, query_indices, Y_query, source):
       #  print(query_indices)
        #  print(self.test_mask)
        #  print(self.unlabeled_mask)
        #  print(self.labeled_mask)
        #  print(np.intersect1d(query_indices, self.unlabeled_mask))
        #  print(len(query_indices))
        #  print(len(np.intersect1d(query_indices, self.unlabeled_mask)))
        #  print(np.setdiff1d(query_indices, self.unlabeled_mask))

        # remove before performance measurements -> only a development safety measure
        #  assert len(np.intersect1d(query_indices, self.labeled_mask)) == 0
        #  assert len(np.intersect1d(query_indices, self.test_mask)) == 0
        #  assert len(np.intersect1d(query_indices, self.unlabeled_mask)) == len(
        #      query_indices
        #  )
        #  print("Label: ", query_indices)

        self.labeled_mask = np.append(self.labeled_mask, query_indices, axis=0)

        for element in query_indices:
            self.unlabeled_mask = self.unlabeled_mask[self.unlabeled_mask != element]

        self.label_source[query_indices] = source
        self.Y[query_indices] = Y_query
        # is not working with initial labels, after that it works, but isn't needed
        #  self.Y[query_indices] = Y_query
        # remove before performance measurements -> only a development safety measure
        #  assert len(np.intersect1d(query_indices, self.unlabeled_mask)) == 0
        #  assert len(np.intersect1d(query_indices, self.test_mask)) == 0
        #  assert len(np.intersect1d(query_indices, self.labeled_mask)) == len(
        #      query_indices
        #  )

    def unlabel_samples(self, query_indices):
        
        self.unlabeled_mask = np.append(self.unlabeled_mask, query_indices, axis=0)
 
        for sample in query_indices:
            self.labeled_mask = self.labeled_mask[self.labeled_mask != sample]

        self.Y[query_indices] = -1
    
    def update_samples(self, query_indices, Y_query):
        self.Y[query_indices] = Y_query
        


    def label_samples(self, query_indices, Y_query, source):
        # remove from train_unlabeled_data and add to train_labeled_data
        self._label_samples_without_clusters(query_indices, Y_query, source)

        # removed clustering code
        # remove indices from all clusters in unlabeled and add to labeled
        #  for cluster_id in self.train_unlabeled_cluster_indices.keys():
        #      list_to_be_removed_and_appended = []
        #      for indice in query_indices:
        #          if indice in self.train_unlabeled_cluster_indices[cluster_id]:
        #              list_to_be_removed_and_appended.append(indice)
        #
        #      # don't change a list you're iterating over!
        #      for indice in list_to_be_removed_and_appended:
        #          self.train_unlabeled_cluster_indices[cluster_id].remove(indice)
        #          self.train_labeled_cluster_indices[cluster_id].append(indice)
        #
        #  # remove possible empty clusters
        #  self.train_unlabeled_cluster_indices = {
        #      k: v for k, v in self.train_unlabeled_cluster_indices.items() if len(v) != 0
        #  }

    def get_true_label(self, query_indice):
        return self.Y[query_indice]
