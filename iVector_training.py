import bob.kaldi
import bob.io.audio
import numpy as np
import glob
import os
import librosa
import soundfile as sf
import math
from collections import defaultdict
from utils import configuration, silence_detection
from scipy.spatial import distance


# loading configuration parameters from config.yaml
config = configuration()


# voxceleb1_mfcc_path = './voxceleb1_spkr_MFCCs/'
voxceleb1_mfcc_path = config['dirs']['mfcc_path_dev_speakerwise']
MFCC_iVec_feats = [np.load(os.path.join(voxceleb1_mfcc_path,npy_file)) for npy_file in os.listdir(voxceleb1_mfcc_path)]


# array dimension manipulation for iVector training
MFCC_iVec_feats = [np.expand_dims(np_array,axis=0) for np_array in MFCC_iVec_feats]


def get_min_array_dim(MFCC_iVec_feats):
    '''
    function returns np array of the samllest dimesion of all np arrays for iVector training
    have to be of ther same dimension
    '''
    min_array_dim = 0

    for i,np_array in enumerate(MFCC_iVec_feats):
        if i == 0:
            min_array_dim = np_array.shape[1]

        else:
            if np_array.shape[1] < min_array_dim:
                min_array_dim = np_array.shape[1]
    return min_array_dim


min_array_dim = get_min_array_dim(MFCC_iVec_feats)
print('min_array_dim',min_array_dim)

MFCC_iVec_feats = [np_array[:,0:min_array_dim,:] for np_array in MFCC_iVec_feats]
MFCC_iVec_feats = np.vstack(MFCC_iVec_feats)
print('MFCC_iVec_feats.shape',MFCC_iVec_feats.shape)

# MFCC_iVec_feats.shape - <no. of speakers,mfcc shape,mfcc dimension>


full_GMM_UBM_model = open(config['model_files']['full_GMM-UBM'],'r').read()

print('training ivector')
ivector_model = bob.kaldi.ivector_train(MFCC_iVec_feats, 
                                        full_GMM_UBM_model,
                                        config['model_files']['iVector'], 
                                        num_gselect=config['iVector_params']['num_gselect'], 
                                        ivector_dim=config['iVector_params']['ivector_dim'], 
                                        use_weights=config['iVector_params']['use_weights'], 
                                        num_iters=config['iVector_params']['num_iters'], 
                                        min_post=config['iVector_params']['min_post'], 
                                        num_samples_for_weights=config['iVector_params']['num_samples_for_weights'],
                                        posterior_scale=config['iVector_params']['posterior_scale'])

print(ivector_model)