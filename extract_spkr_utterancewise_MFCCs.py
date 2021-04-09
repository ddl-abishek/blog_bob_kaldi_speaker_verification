import numpy as np
import os
import bob.io.audio
import bob.kaldi
from multiprocessing import Pool
from utils import configuration, silence_detection
import argparse

# loading configuration parameters from config.yaml
config = configuration()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="select dataset 'dev' or 'test'")
args = parser.parse_args()

# for extracting MFCCs in the dataset enmtered as argument set, 
voxceleb1_path = config['dirs'][f'voxceleb1_path_{args.dataset}']


def extract_saveMFCCs(spkr,voxceleb1_path = config['dirs'][f'voxceleb1_path_{args.dataset}']):
    
    spkr_path = os.path.join(voxceleb1_path,spkr)
    print("\n\n",spkr)

    for utterance in os.listdir(spkr_path):
        if utterance[-4:] == '.wav': #checking if .wav file 
            print(utterance)
            utterance_path = os.path.join(spkr_path,utterance)
            data = bob.io.audio.reader(utterance_path)
            
            speech_segs = silence_detection(data)
            norm_speech = data.load()[0]/2**15
            
            mfcc = []
            for seg in speech_segs:
                seg_mfcc = bob.kaldi.mfcc(norm_speech[seg[0]:seg[1]])
                mfcc.append(seg_mfcc)
                
            mfcc = np.vstack(mfcc)
            
            spkr_save_path = os.path.join(config['dirs'][f'mfcc_path_{args.dataset}'],spkr)
            if not(os.path.isdir(spkr_save_path)): #create directory if it does not exist
                os.makedirs(spkr_save_path)

            mfcc_path = f'{spkr_save_path}/' + utterance.split('.')[0] + '_mfcc.npy'
            np.save(mfcc_path,mfcc)

# the bigger model is trained on first 100 speakers of VoxCeleb1/dev
# spkr_list = [spkr for spkr_index,spkr in enumerate(os.listdir(voxceleb1_path)) if spkr_index<100]

# for mini model all 5 speakers from ./VoxCeleb1_mini are used 
spkr_list = [spkr for spkr_index,spkr in enumerate(os.listdir(voxceleb1_path)) ]


with Pool(10) as p:
    p.map(extract_saveMFCCs,spkr_list)