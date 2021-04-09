import numpy as np
import os
import bob.io.audio
import bob.kaldi
from multiprocessing import Pool
from collections import defaultdict
from utils import configuration, silence_detection
import argparse

# loading configuration parameters from config.yaml
config = configuration()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="select dataset 'dev' or 'test'")
args = parser.parse_args()

spkr_mfccs_path = config['dirs'][f'mfcc_path_{args.dataset}']
spkrs_mfcc = [spkr for spkr in os.listdir(spkr_mfccs_path)]

def extract_save_iVectors(spkr,
                          spkr_mfccs_path = config['dirs'][f'mfcc_path_{args.dataset}']):
    '''
    Extract iVectors from each individual utterance's mfcc and save it in the respective speaker's directory.
    
    Each mfcc is of the format
    <channel>_<utterance no.>_mfcc.npy 
    e.g.: 1zcIwhmdeo4_00002_mfcc.npy
    Here, 
        '1zcIwhmdeo4' is the channel
        '00002' is the utternace number 
    Each channel may hav multiple utterances like
    1zcIwhmdeo4_00001_mfcc.npy, 1zcIwhmdeo4_00002_mfcc.npy, 1zcIwhmdeo4_00003_mfcc.npy
    
    Only one utterance per channel is chosen
    
    Parameters:
    spkr(str) - speaker whose mfccs will be used to extract iVectors
    spkr_mfccs_path(str) - path where mfccs ofall speaker are stored. 
                           Each directory in this path represents a speaker's mfccs
    Returns:
    '''
        
    if not(os.path.isdir(config['dirs'][f'iVector_path_{args.dataset}'])):
        os.makedirs(config['dirs'][f'iVector_path_{args.dataset}'])
        
    spkr_path = os.path.join(spkr_mfccs_path,spkr)

    spkr_save_path = f"{config['dirs'][f'iVector_path_{args.dataset}']}{spkr}/" #path where speaker's iVector is saved
        
    fubm = open(config['model_files']['full_GMM-UBM']).read()
    ivector_extractor = open(config['model_files']['iVector']).read()

    channel_dict = defaultdict(str) # dictionary to keep track of each individual channel

    for mfcc in os.listdir(spkr_path):
        try:
            if not(channel_dict[mfcc.split('_')[0]]): # extract ivector only if the channel does not exist 

                channel_dict[mfcc.split('_')[0]] = True # creating the channel 

                mfcc_path = os.path.join(spkr_path,mfcc)
                mfcc_feat = np.load(mfcc_path)
                
                if mfcc_feat.shape[0] > 2000+1: # utterances longer than 30s are trimmed to 30s
                    mfcc_feat = mfcc_feat[0,2000+1,:]
                    
                ivector = bob.kaldi.ivector_extract(mfcc_feat, 
                                                    fubm,
                                                    ivector_extractor,
                                                    num_gselect=config['iVector_extract_params']['num_gselect'],
                                                    min_post=config['iVector_extract_params']['min_post'],
                                                    posterior_scale=config['iVector_extract_params']['posterior_scale'])

                if not(os.path.isdir(spkr_save_path)):
                    os.makedirs(spkr_save_path)

                ivector_save_path = f'{spkr_save_path}' + mfcc.split('_mfcc')[0] + '_ivec.npy'
                print(ivector_save_path)
                np.save(ivector_save_path,ivector)
        except:
            continue

for i in range(len(spkrs_mfcc)):
    extract_save_iVectors(spkrs_mfcc[i])

