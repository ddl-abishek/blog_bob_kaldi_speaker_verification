from flask import Flask, request, jsonify
import base64
import tempfile
import os
import bob
from bob.io.audio import reader
import bob.kaldi
from utils import configuration, silence_detection
import numpy as np
from datetime import datetime
from scipy.spatial import distance

config = configuration()

class iVector_model:
    def __init__(self,data):
        self.data = data
        self.fubm = open(config['model_files']['full_GMM-UBM']).read()
        self.ivector_extractor = open(config['model_files']['iVector']).read()

    def get_mfcc(self):
        norm_speech = self.data.load()[0]/2**15
        speech_segs = silence_detection(self.data)
        
        mfcc = []
        for seg in speech_segs:
            seg_mfcc = bob.kaldi.mfcc(norm_speech[seg[0]:seg[1]])
            mfcc.append(seg_mfcc)

        mfcc = np.vstack(mfcc)
        return mfcc
        
    def get_iVector(self):
        mfcc = self.get_mfcc()
        if mfcc.shape[0] > 2000+1: # utterances longer than 30s are trimmed to 30s
            mfcc = mfcc[0,2000+1,:]
        
        iVector = bob.kaldi.ivector_extract(mfcc,
                                            self.fubm,
                                            self.ivector_extractor,
                                            num_gselect=config['iVector_extract_params']['num_gselect'],
                                            min_post=config['iVector_extract_params']['min_post'],
                                            posterior_scale=config['iVector_extract_params']['posterior_scale'])
        return iVector

    def enroll(self,spkr_id):
        if not(spkr_id in os.listdir(config['enrolled_speakers']['path'])):
            iVector = self.get_iVector()
            
            os.makedirs(os.path.join(config['enrolled_speakers']['path'],spkr_id))
                
            spkr_iVec_path = os.path.join(config['enrolled_speakers']['path'],spkr_id,
                                          f'{spkr_id}_{str(datetime.now())}_iVec.npy')
            np.save(spkr_iVec_path,iVector)
            return True
            
        else:
            return False

    def verify(self,spkr_id):
        if spkr_id in os.listdir(config['enrolled_speakers']['path']):
            spkr_path = os.path.join(config['enrolled_speakers']['path'],spkr_id) 
            spkr_iVec_path = os.path.join(spkr_path,os.listdir(spkr_path)[0])
            enroll_iVector = np.load(spkr_iVec_path)
            verify_iVector = self.get_iVector()
                    
            return 1-distance.cosine(enroll_iVector,verify_iVector)
        else:
            return False

def decode_base64(base64string):
    b64_byte = base64.b64decode(base64string)
    wav_temp = tempfile.TemporaryFile()
    
    with open("./" + str(wav_temp.name) ,'wb') as wav: 
        wav.write(b64_byte)
    
    data = bob.io.audio.reader(str(wav_temp.name))
    
    if data.rate!=16000:
        return {'status':'fail',
                'msg':'acceptable sampling rate is 16kHz',
                'wav_temp' :wav_temp, 
                'data' : data}

    elif data.load().shape[0] != 1:
        return {'status' : 'fail',
                'msg' : 'mono channel audio required',
                'wav_temp' : wav_temp,
                'data' : data}
    

    return {'status': 'pass',
            'msg' : 'acceptable audio',
            'wav_temp' : wav_temp,
            'data' : data}

def verifySpeaker(payload):
    [task,aud_b64,spkr_id] = payload.values()
    [status,msg,wav_temp,data] = decode_base64(base64string).values()
    os.unlink("./" + str(wav_temp.name) )
    
    if status == 'fail':
        return {'status': status,'msg' : msg}
    
    if task == 'enroll':
        model = iVector_model(data)
        enroll_status = model.enroll(spkr_id)
        
        if enroll_status:
            return {'status' : 'pass','msg' : f'speaker {spkr_id} enrolled'}
        else:
            return {'status': 'fail', 'msg' : f'speaker {spkr_id} already exists'}
    
    elif task == 'verify':
        model = iVector_model(data)
        verify_status = model.verify(spkr_id)
        
        if not(verify_status):
            return {'status' : 'fail' , 'msg' : f'speaker {spkr_id} does not exist'}
        else:
            return {'status' : 'pass' , 'msg' : f'cosine similarity score {verify_status}'}
    
    else:
        return {'status' : 'fail', 'msg' : 'enter valid task "enroll" or "verify"'}
    

# if __name__ == "__main__":    
#     base64string = open('./base64_VoxCeleb1_mini/id10001_1zcIwhmdeo4_00001.txt','r').read()
#     payload =  { 'task' : 'verify',
#                  'aud_b64' : base64string,
#                  'spkr_id' : 'r12079'}

#     print(verifySpeaker(payload))