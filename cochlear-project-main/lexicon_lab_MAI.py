# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:16:33 2023

@author: Mai Al-Khatib
"""
import numpy as np
import pandas as pd
import switch
from sklearn.metrics.pairwise import cosine_similarity as sklearnCosineSim
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

embeddings_word2vec = {}
# C:\\BowdoinCodingChallenge\\cochlear-project-main\\cochlear-project-main\\
with open("word2vec.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_word2vec[word] = vector
embeddings_word2vec["modName"] = "word2vec"    
        
# C:\\BowdoinCodingChallenge\\cochlear-project-main\\cochlear-project-main\\
embeddings_speech2vec = {}
with open("speech2vec.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_speech2vec[word] = vector
embeddings_speech2vec["modName"] = "speech2vec"    

#global pairwiseData
#global clusterData 


class similarity:
    
    def cosine_similarity(self, word1, word2, model):
        if not(word1 in model) or not(word2 in model):
            return -2
        return np.dot(model[word1], model[word2])/(np.linalg.norm(model[word1])* np.linalg.norm(model[word2]))
    
    def pairwise_similarity(self, data, model):
        word = ""
        pid = "0"
        entries = []
        with open(data) as cdata:
            for line in cdata:
                sim  = 2
                dataPoint = line.split()
                if pid == dataPoint[0]:
                    sim = self.cosine_similarity(word, dataPoint[1], model)
                pid = dataPoint[0]
                word = dataPoint[1]
                entry = (pid, word, sim, model["modName"])
                entries.append(entry)
        
                
        pairwisedf = pd.DataFrame(entries)
        pairwisedf.columns = ['ID', 'animal', 'sim', 'modName']
        return pairwisedf
    
class clusters:

    def compute_clusters(self, data, model):
        s = similarity()
        dfPairwise = s.pairwise_similarity(data, model)
        sd = switch.switch_simdrop(dfPairwise['animal'], dfPairwise['sim'])
        dfPairwise['switch'] = sd
        fluencyListEmbeddings = {}
        for animal in dfPairwise['animal'].unique():
            if animal in model:
                fluencyListEmbeddings[animal]= model[animal]
            else:
                fluencyListEmbeddings[animal] = [0] * 50
        
        fldf = pd.DataFrame(fluencyListEmbeddings).T
        #print("columns: "+ ", ".join(fldf.columns))
        new_df = fldf.drop(columns= 0)
        df_cosine = pd.DataFrame(sklearnCosineSim(new_df,dense_output=True))
        # kmeans elbow function to create the optimal clusters
        cs = []
        plt.figure(figsize=(10,6))
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
            kmeans.fit(df_cosine)
            cs.append(kmeans.inertia_)
        plt.plot(range(1, 11), cs)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('CS')
        plt.show()
        k = input("enter number of clusters \t")
        data = df_cosine
        pca = PCA(2)
        transform = pca.fit_transform(data)
        kmeans = KMeans(n_clusters= int(k))
        label = kmeans.fit_predict(transform)
        u_labels = np.unique(label)
        
        # list of data point participant ID, animal name, and cluster
        csData={'animal': list(fldf.index), 'cluster':label, 'modName':model['modName']}
                
        return pd.DataFrame(csData), dfPairwise

    def visualize_clusters(self, ID, cData, pwData):
        
        if len(cData) == 0 or len(pwData) == 0:
            print("No data to visualize")
            return
        participantData = pwData.loc[pwData['ID'] == ID]
        modelSummaries = pd.DataFrame(columns=['ID', 'clusterMean', 'switchMean', 'modName'])
        for model in np.unique(cData['modName']):
            clusterModelData = cData.loc[cData['modName']==model]
            uClusters = np.unique(clusterModelData['cluster'])            
            clusterCounts = {}
            for c in uClusters:
                clusterCounts[c]= len(clusterModelData.loc[clusterModelData['cluster']==c])
            
            clusterMean = sum(clusterCounts.values())/ len(clusterCounts)
            pairwiseModelData = participantData.loc[participantData['modName'] == model]
            
            switchCounts = {}
            for s in np.unique(pairwiseModelData['switch']):
                switchCounts[s] = len(pairwiseModelData.loc[pairwiseModelData['switch']==s])
            switchMean = sum(switchCounts.values())/ len(switchCounts)
            row = [[ID, clusterMean, switchMean, model]]
            sumDF = pd.DataFrame(row, columns=['ID', 'clusterMean', 'switchMean', 'modName'])
            if len(modelSummaries)>0:
                modelSummaries = pd.concat((sumDF, modelSummaries), axis=0)
            else:
                modelSummaries = sumDF
        return modelSummaries

#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
#from gensim.test.utils import datapath
#from gensim.scripts.glove2word2vec import glove2word2vec

#cosine_similarity = np.dot(embeddings_dict['said'], embeddings_dict['could'])/(np.linalg.norm(embeddings_dict['said'])* np.linalg.norm(embeddings_dict['could']))
#cosine_similarity2 = np.dot(embeddings_dict['said'], embeddings_dict['said'])/(np.linalg.norm(embeddings_dict['said'])* np.linalg.norm(embeddings_dict['said']))

#wv_from_text = KeyedVectors.load_word2vec_format(datapath('C:\\BowdoinCodingChallenge\\cochlear-project-main\\cochlear-project-main\\word2vec.txt'), binary=False)
#wv = KeyedVectors.load("word2vec.txt", mmap='r')

# modelW = Word2Vec.load_word2vec_format('C:\\BowdoinCodingChallenge\\cochlear-project-main\\cochlear-project-main\\word2vec.txt', binary=False)
#modelW = KeyedVectors.load_word2vec_format('C:\\BowdoinCodingChallenge\\cochlear-project-main\\cochlear-project-main\\word2vec.txt', binary=False)

#from scipy import spatial
    