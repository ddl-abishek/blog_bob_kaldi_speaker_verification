import bob.kaldi
import bob.io.audio
import numpy as np
import glob
import os
import librosa
import soundfile as sf
import math
from collections import defaultdict
import tempfile
from scipy.spatial import distance
from utils import configuration, silence_detection

# loading configuration parameters from config.yaml
config = configuration()


#function to select specific utterances per speaker to maximize channel variabiity

def utterance_selection(spkr_path,max_utterances = 16):
    '''
    given a speaker's directory of utterances, the function retuens a list of selected utterances 
    (=max_utterances) that maximize the channel variability
    '''
    utter_dict = defaultdict(list) # {'channel_name' : [list of utterances in that channel]}
    
    
    for utterance in os.listdir(spkr_path):
        if utterance[-4:] == '.wav':
            utter_dict[utterance.split('_')[0]].append(utterance)

    utter_count = 0
    channel_index = 0
    selected_utterances = []

    while utter_count < max_utterances:
        for channel in utter_dict.keys():
            if utter_count < max_utterances:
                try:
                    selected_utterances.append(os.path.join(spkr_path,utter_dict[channel][channel_index]))
                    utter_count += 1
                except:# some channels may have more utterances than others
                    continue
            else:
                break
        channel_index += 1

    return selected_utterances


#### function to build and save MFCC array for each speaker


def save_speaker_MFCC(dataset_path,max_utter_duration=5):
    '''
    function builds a big MFCC array by extracting mfccs from several speakers' utterances where
    each utterance is trimmed till ~5 seconds of speech activity
    '''
    
    for spkr_index,spkr in enumerate(os.listdir(dataset_path)):
        MFCC_feats = []
        if spkr_index >= 100: # UBM built on only 100 speakers worth of data; break when limit reached
            break
            
        else:
            print('--------------------------------------------------------------------------------------')
            print(f'analyzing utterances from speaker {spkr}')
            spkr_path = os.path.join(dataset_path,spkr)
            
            # utterances selected based on maximing channel variability
            selected_utterances = utterance_selection(spkr_path)
            
            for utterance in selected_utterances:
                wav = utterance.split('/')[-1]
                print(f'  utterance {wav}')

                data = bob.io.audio.reader(utterance)
                norm_speech = data.load()[0]/2**(data.bits_per_sample-1) # normalizing speech utterance
                
                speech_segs = silence_detection(data)
                
                utter_duration = 0
                
                for seg in speech_segs:
                    if utter_duration <= max_utter_duration:
                        seg_duration = (seg[1] - seg[0])/data.rate
                        
                        if utter_duration + seg_duration > max_utter_duration:
                            upper_bound = int(5*data.rate + seg[0])

                        elif utter_duration + seg_duration <= max_utter_duration:
                            upper_bound = seg[1]
                            
                        target_speech_seg = norm_speech[seg[0]:upper_bound+1]

                        mfcc_seg = bob.kaldi.mfcc(target_speech_seg)
                        MFCC_feats.append(mfcc_seg)

                        utter_duration += (upper_bound - seg[0])/data.rate
                        
        print(f' saving {spkr}.npy')
        if not(os.path.isdir(config['dirs']['mfcc_path_dev_speakerwise'])): #create directory if it does not exist
            os.makedirs(config['dirs']['mfcc_path_dev_speakerwise'])

        np.save(f"{config['dirs']['mfcc_path_dev_speakerwise']}{spkr}.npy",np.vstack(MFCC_feats))
                        
    return 1


# driver code

save_speaker_MFCC(config['dirs']['voxceleb1_path_dev'])

# extracting all speakers MFCC files (npy files) and stacking them for iVector training


# MFCC_feats = [np.load('./voxceleb1_spkr_MFCCs/'+np_file) for np_file in os.listdir('./voxceleb1_spkr_MFCCs/')]

MFCC_feats = [np.load(f"{config['dirs']['mfcc_path_dev_speakerwise']}/{np_file}") for np_file in os.listdir(config['dirs']['mfcc_path_dev_speakerwise'])]


MFCC_feats = np.vstack(MFCC_feats)


print(MFCC_feats.shape)

### UBM training
print('training diagonal UBM')
diag_GMM_UBM_file = config['model_files']['diag_GMM-UBM']

diag_GMM_UBM_model = bob.kaldi.ubm_train(MFCC_feats,
                                         config['model_files']['diag_GMM-UBM'],
                                         num_threads=config['diag_gmm_ubm_params']['num_threads'],
                                         min_gaussian_weight=config['diag_gmm_ubm_params']['min_gaussian_weight'],
                                         num_gauss=config['diag_gmm_ubm_params']['num_gauss'],
                                         num_gauss_init=config['diag_gmm_ubm_params']['num_gauss_init'],
                                         num_gselect=config['diag_gmm_ubm_params']['num_gselect'],
                                         num_iters_init=config['diag_gmm_ubm_params']['num_iters_init'],
                                         num_iters=config['diag_gmm_ubm_params']['num_iters'],
                                         remove_low_count_gaussians=config['diag_gmm_ubm_params']['remove_low_count_gaussians'])


print(diag_GMM_UBM_model)
print('training full covariance UBM')

full_GMM_UBM_model = bob.kaldi.ubm_full_train(MFCC_feats, 
                                              diag_GMM_UBM_model, 
                                              config['model_files']['full_GMM-UBM'], 
                                              num_gselect=config['full_gmm_ubm_params']['num_gselect'], 
                                              num_iters=config['full_gmm_ubm_params']['num_iters'], 
                                              min_gaussian_weight=config['full_gmm_ubm_params']['min_gaussian_weight'])


print(full_GMM_UBM_model)