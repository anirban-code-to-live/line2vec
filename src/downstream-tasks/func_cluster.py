import os
from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd
import time
import random

import itertools

def cluster(dataset,n_times=1):
    
    path = '../data/' + dataset
    cluster_dict = {'citeseer':6, 'pubmed':3, 'MSA':3, 'Wiki':4, 'cora':7, 'Flickr':9}
    n_clusters = cluster_dict[dataset]
    print 'number of clusters is ',n_clusters

    ##################################
    
    label = pd.read_csv(path+'/label.csv',header=None)
    print 'label shape is ',label.shape
    
    ct = label.shape[0]
    file_lst = ['link_structure_embedding'] #,'ref_embeddings','cont_embeddings']
    res = np.zeros((len(file_lst)))
    
    for epoch in range(n_times):
        rand_index = random.sample(np.arange(0, ct), int(ct*1))  #taking 5% data
        #the exapmples are same for all embedddings, so no partiality
        

        for i in range(len(file_lst)):
            
            filetype = file_lst[i]
            print '\n\n#########',filetype

            points = pd.read_csv(path+'/'+filetype+'.csv',header=None,sep=' ')
            print 'data shape is ',points.shape

            points = points.iloc[rand_index]
            true_labels = list(label.iloc[rand_index][0])

            print 'sampled data shape ',points.shape,len(true_labels)


            ###############################
#             st = time.time()
            spectral = KMeans( n_clusters, n_jobs=-1 ).fit(points)
#             end = time.time()
#             print 'time taken for spectral on '+filetype+'  is '+str(end-st)



            clusLabels =list(spectral.labels_)

            print len(true_labels)
            print len(clusLabels)
            clusLabels=map(float,clusLabels)
            true_labels=map(float,true_labels)

            ###############################
            permutations = list(itertools.permutations(range(n_clusters),n_clusters))
            accuracies = []
            clusLabels_mapped = [-1]*len(clusLabels)

            for match in permutations:
                for l in range(len(clusLabels)):
                    clusLabels_mapped[l] = match[int(clusLabels[l])]
                accuracies.append(sum(1 for a,b in zip(clusLabels_mapped,true_labels) if a == b) / float(len(clusLabels)))

            accFinal = max(accuracies)
            res[i] = res[i]+accFinal
            # match_final = permutations[np.argmax(np.asarray(accuracies, dtype=np.float32))]
            print 'Accuracy:'+filetype+'   '+str(accFinal)
        print 'sampled result uptill now is \n',res/(1+epoch),'\n\n'
    print dataset,'final result',res/n_times



