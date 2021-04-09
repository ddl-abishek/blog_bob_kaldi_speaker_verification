To run the project, follow the steps below

1. Activate the environment
    source activate bob_kaldi
2. Run the command 
    download_dependencies.py 
    to download the mini-dataset VoxCeleb1_mini.tar.gz and a python dependency/library pyAudioAnalysis.tar.gz
    
3. Untar the files
    tar -xvzf pyAudioAnalysis.tar.gz
    tar -xvzf VoxCeleb1_mini.tar.gz

3. Install ipykernel (optional)
    conda install ipykernel

4. Create kernel for kaldi (optional)
    python -m ipykernel install --user --name bob_kaldi --display-name "Python3 (kaldi)"
    (Whenever you are using any notebook, make sure to this kernel)
    
5. cd /pyAudioAnalysis
   pip install -e .

6. Extract MFCCs from the dev dataset
    python extract_spkr_utterancewise_MFCCs.py --dataset dev

   Extract MFCCs from the test dataset
    python extract_spkr_utterancewise_MFCCs.py --dataset test
    
7. Train UBM
    python UBM_training.py

8. Train iVector
    python iVector_training.py
    
9. Extract iVectors
    python extract_spkr_utterancewise_iVectors.py --dataset dev
    python extract_spkr_utterancewise_iVectors.py --dataset test