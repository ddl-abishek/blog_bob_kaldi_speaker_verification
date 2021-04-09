import os
from pyAudioAnalysis import audioSegmentation as aS
import confuse

def configuration():
	'''
	Returns configuration dictionary
	'''
	return confuse.Configuration(os.getcwd(),__name__).get()

def silence_detection(data):
    '''
    Returns list of lists containing the start and end time stamps of speech segments
    e.g.: [[0.05,0.07],[0.09,0.012]] --> speech exists in the intervals 0.05-0.07 and 
                                         0.09-1.12   
    Utlises a pretained Hidden Markov Model in pyAudioanalysis that determines speech 
    activity.

    Parameters: 
    data (reader): reader returned by bob.io.audio.reader  
  
    Returns: 
    intervals(list): list of lists containing intervals where speech segments exist
    '''
    config = configuration()
    intervals = aS.silence_removal(data.load()[0],
                                   config['speech_params']['sample_rate'],
                                   config['speech_params']['frame_length'],
                                   config['speech_params']['frame_overlap'],
                                   smooth_window = config['speech_params']['smooth_window'],
                                   weight = config['speech_params']['sil_rem_wt'],
                                   plot = config['speech_params']['sil_rem_plt'])

    for i in range(len(intervals)):
        intervals[i] = [int(stamp*config['speech_params']['sample_rate']) for stamp in intervals[i]]

    return intervals