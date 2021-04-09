'''
Algo:

Let the number of speakers be N (=100 in test set ; also 100 in training set) 
Let the variable to represent a speaker be x; xij, where 0<=i<=N and 0<=j<=Mi ;
Mi is total no. of utterances speaker i

Each speaker has a certain number of speech files. CSS score will be evaluated between each combination 
of these utterances i.e. Mi-C2

Then any one utterance will be selected from Mi speech utterances and the CSS will be evaluated against 
each speaker's random utterance

A positive event is when there is a match between the enrolled speaker and the speaker who wishes to be verified

A negative evernt is when there is a mismatch between the enrolled speaker and the speaker who wishes to be verified
'''

import bob.kaldi
import bob.io.audio
import numpy as np
import glob
import os
import librosa
from pyAudioAnalysis import audioSegmentation as aS
import soundfile as sf
import math
from collections import defaultdict
from itertools import combinations
import tempfile
from scipy.spatial import distance
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from utils import configuration, silence_detection
from sklearn.metrics import roc_curve, precision_recall_curve

def css(ivec_a,ivec_b):
    '''
    Returns the cosine similarity score of 2 ivectors
    '''   
    return 1-distance.cosine(ivec_a,ivec_b)

# voxceleb1_spkrs_ivecs_test is a dictionary of a list numpy arrays where each key is the speaker name/id 
# and the corresponding value is a list of ivectors 

# Also note that only ivectors with the string 'enroll' present will be used for evaluating TPR|FPR

voxceleb1_spkrs_ivecs_test = defaultdict(dict)

# import pdb; pdb.set_trace();

# #### path where all iVectors of each speaker are present

config = configuration()

# spkrs_ivecs_test_path = './voxceleb1_utterancewise_spkr_iVectors_test_combining_utterances/'
spkrs_ivecs_test_path = config['dirs']['iVector_path_test']


# # iterating over each speaker to load all the ivectors 

for spkr in os.listdir(spkrs_ivecs_test_path):
    spkr_path = os.path.join(spkrs_ivecs_test_path,spkr)
    
    voxceleb1_spkrs_ivecs_test[spkr] = {}
    
    idx = 0 #speaker session of ivector index
    for ivec in os.listdir(spkr_path):
        ivec_path = os.path.join(spkr_path,ivec)
        voxceleb1_spkrs_ivecs_test[spkr][idx] = {'file':ivec,'ivec':np.load(ivec_path)}
        idx += 1

print('writing pickle file of voxceleb1_mini_spkrs_ivecs_test.pkl')
pickle.dump( voxceleb1_spkrs_ivecs_test, open( "./voxceleb1_mini_spkrs_ivecs_test.pkl", "wb" ) )


# voxceleb1_spkrs_ivecs_test = pickle.load(open("./voxceleb1_mini_spkrs_ivecs_test.pkl","rb"))


#  confusion matrix evaluation 
# choosing np.ones because cosine similaroty is a value between -1 and 1
total_ivecs = 0
for spkr in voxceleb1_spkrs_ivecs_test.keys():
    total_ivecs += len(voxceleb1_spkrs_ivecs_test[spkr].keys())
    
print("total number of ivectors ",total_ivecs)

confusion_mat = np.empty((total_ivecs,total_ivecs))
confusion_mat[:] = np.nan
print("confusion_mat.shape",confusion_mat.shape)

all_spkrs_ivecs = []

for spkr in voxceleb1_spkrs_ivecs_test.keys():
    for sess_idx in voxceleb1_spkrs_ivecs_test[spkr].keys():
        all_spkrs_ivecs.append(voxceleb1_spkrs_ivecs_test[spkr][sess_idx]['ivec'])


for i in range(len(all_spkrs_ivecs)):
    css_sect = []
    for j in range(i,len(all_spkrs_ivecs)):
        css_sect.append(css(all_spkrs_ivecs[i],all_spkrs_ivecs[j]))
        
    confusion_mat[i,i:] = css_sect
    confusion_mat[i:,i] = css_sect

pandas_index = []
pandas_cols = []

for spkr in voxceleb1_spkrs_ivecs_test.keys():
    
    for sess_idx in voxceleb1_spkrs_ivecs_test[spkr].keys():
        pandas_index.append(spkr+'_'+str(sess_idx))

pandas_cols = pandas_index

confusion_df = pd.DataFrame(confusion_mat,index=pandas_index,columns=pandas_cols)

# confusion_df


y_pred = []
y_actual = []

for col_spkr_uttr in confusion_df.columns:
    for cos_sim_score,row_spkr_uttr in zip(confusion_df[col_spkr_uttr],pandas_index):
        if col_spkr_uttr.split('_')[0] == row_spkr_uttr.split('_')[0]: # if the utterance is from the same speaker
            # then y_actual is 'one' else 'zero'
            y_actual.append(1)
        else:
            y_actual.append(0)
                
        # if cos_sim_score is >0.75 then speakers match else they do not
        if cos_sim_score > 0.75:
            y_pred.append(1)
        else:
            y_pred.append(0)
            
print('classification report')
print(classification_report(y_actual,y_pred))


fpr = 0
for idx,label in enumerate(y_actual):
    if label == 0 and y_pred[idx] == 1:
        fpr += 1

fpr /= len(y_pred)
print("False Positive Rate",fpr)
print("False Positive Rate * 100",fpr*100)


fnr = 0
for idx, label in enumerate(y_actual):
    if label == 1 and y_pred[idx] == 0:
        fnr += 1

fnr /= len(y_pred)
print("False Negative Rate",fnr)
print("False Negative Rate * 100",fnr*100)