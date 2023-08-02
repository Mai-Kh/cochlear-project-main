# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 20:21:52 2023

@author: Mai Al-Khatib
"""

import lexicon_lab_MAI as ll

simExample = ll.similarity()
simExample.cosine_similarity('cat', 'lizard', ll.embeddings_speech2vec)
len(simExample.pairwise_similarity("data-cochlear.txt", ll.embeddings_word2vec))
c = ll.clusters()
w2vClusters, w2vData = c.compute_clusters("data-cochlear.txt", ll.embeddings_word2vec)
s2vClusters, s2vData = c.compute_clusters("data-cochlear.txt", ll.embeddings_speech2vec)
clusterData = ll.pd.concat((w2vClusters, s2vClusters), axis=0)
pwData = ll.pd.concat((w2vData, s2vData), axis=0)
sums = c.visualize_clusters('SOQ-329', clusterData, pwData)
sums.plot.bar(x='modName', y = ['clusterMean', 'switchMean'])