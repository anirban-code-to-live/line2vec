
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# In[3]:


def classify(dataset, fname):
	path = '../../data/' + dataset
	print path

	print '\n\n ',fname
	df = pd.read_csv(path + '/' + fname+'.csv',sep=' ',header=None)
	print df.shape

	###########################

	label = pd.read_csv(path+'/edge_labels.csv',header=None)
	print label.shape

	df['label'] = label
	#print df.shape
	#############################

	#shufling the whole of data with corresponding labels
	#         print dataset
	df = df.sample(frac=1)
	df = df.reset_index(drop=True)

	X = df.drop('label',axis=1)   #all columns except 'label' are the features for training  #drop isn't inplace
	y = df['label']

	avg_macro = []; avg_micro = []; std_micro = []; std_macro = []; avg_weighted_f1 = []; std_weighted_f1 = [];
	n_times = 10

	train_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  #training set size (in %) used for training.
	for per in train_percentage:
	    #print '.',
	    macro = []   #reset for each percentage
	    micro = []
	    weighted = []
	    for _ in np.arange(n_times):

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-per)  #don't fix the random number :P  otherwise no use of doing it 10 times
		model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)

		macro.append(f1_score(y_test, y_pred, average='macro'))
		micro.append(f1_score(y_test, y_pred, average='micro'))
                weighted.append(f1_score(y_test, y_pred, average='weighted'))

	    #print per, 'mean is ',np.mean(macro), np.mean(micro)
	    avg_macro.append(np.mean(macro))
	    avg_micro.append(np.mean(micro))
            avg_weighted_f1.append(np.mean(weighted))

	    std_macro.append(np.std(macro))
	    std_micro.append(np.std(micro))
            std_weighted_f1.append(np.std(weighted))


	print 'MACRO',avg_macro
	print '\nMICRO',avg_micro
        print '\nWEIGHTED', avg_weighted_f1
