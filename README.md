# Steps to run the project

### 1. Create the necessary environment in Domino (if it does not already exist) 
To check if the environment already exists, go to the domino deployment and click on Environments in the left most panel. 
In the search, type bob_kaldi_lite and hit enter/return. If bob_kaldi_lite appears in the drop down, the environment is present

If the environment is not present, then follow the steps below to create the environment :-
 - Click on Environments in the left most panel followed by **+Create Environment**
 - Under name type **bob_kaldi_lite**
 - Under description type light version of bob_kaldi (the description can be left blank)
 - Under Base Image, select the Custom Image radio button and copy and paste this URI

        dominodatalab/base:DMD_py3.6_r3.6_2019q4

   (This is the URI of the Domino Minimal Distribution which is lightweight and helps in 
   faster launch times of workspaces)
 - Under Visibility, select the **Globally Accessible** radio button
 - Click on the **+Create Envioronment** button in the bottom of ther dialog. You will be directed to the Overview panel of the environment
 - Click on Edit Dockerfile under Dockerfile Instructions. Type the following:-

        RUN \
          wget https://dsp-workflow.s3-us-west-2.amazonaws.com/bob_kaldi_environment.yml && \
          conda env create -f bob_kaldi_environment.yml
          
    (The above command downloads the conda environment yaml file and creates an environment called bob_kaldi36 in the domino deployment https://try.dominodatalab.com or whatever deployment you may be running this project on)
    
 - Under Pluggable Workspace Tools, type copy and paste the following
 
        jupyterlab:
          title: "JupyterLab"
          iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
          start: [  /var/opt/workspaces/Jupyterlab/start.sh ]
          httpProxy:
            internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
            port: 8888
            rewrite: false
            requireSubdomain: false
          supportedFileExtensions: [ ".ipynb" ]
    (This creates a jupyterlab workspace tool where we will run the project)
 -  Click on the **Build button** at the bottom to create the environment. This build may take a while. Please resume the remaining steps while this build is in progress.

### 2. Clone this git repo to your local system. In your terminal, type 
        git clone https://github.com/ddl-abishek/blog_bob_kaldi_speaker_verification.git
        
###### If this does not work in try.dominodatalab.com, please try the below command.
        wget https://dsp-workflow.s3-us-west-2.amazonaws.com/blog_bob_kaldi_speaker_verification/blog_bob_kaldi_speaker_verification-master.zip
###### Now unzip the file
        unzip ./blog_bob_kaldi_speaker_verification-master.zip

        

### 3. Let's create a project in the deployment. 
 - Navigate to the deployment(https://try.dominodatalab.com or whatever deployment you may be using) and click on Projects in the left most tab.

 - Click on the **New Project** button in the top-right corner. Under Project Name, type **blog_bob_kaldi_speaker_verification**. Under Project Visibility, choose the **Private** radio button. Under Code Repository, choose the **DFS** radio button. Click on **Create Project**. 

### 4. Upload the project files
 - You should be directed to the Overview section of the project. 
 - Click on browse for files. Navigate to the directory where the repo was cloned and select all the repo files.
 - Now click on the **Upload** button.

### 5. Launch a workspace to run the project
 - Click on Workspaces in the left panel. Click on +Click New Workspace.
 - Under **Workspace Name**, give a suitable name to the workspace or leave it blank
 - Under **Workspace Environment**, type **bob_kaldi_lite** and select this environment from the dropdown
 - Under **Workspace IDE**, click on JupyterLab
 - Under **Hardware Tier**, select **Small** from the dropdown
 - Click on Launch Now and wait for the workspace to launch.


### 6. Download dependencies and upload to project
 - In you laptop/computer, run these commands in the terminal and downlaod the dependencies.
 
 - Download a mini (subset of VoxCeleb1) dataset. Refer to the link for more information on the dataset (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
 
        wget https://dsp-workflow.s3-us-west-2.amazonaws.com/VoxCeleb1_mini.tar.gz

   Download pyAudioAnalysis (This dependency/library is used to detect silence segments in speech .wav files)

        wget https://dsp-workflow.s3-us-west-2.amazonaws.com/pyAudioAnalysis.tar.gz
        
 - Now upload these dependencies to the domino workspace by going to the domino worksapce and clicking on the upload button (upward arrow on top of a horizontal line). Navigate to the directory where these dependeencies were saved and upload them. Start a terminal in the domino workspace by clicking on File -> New -> Terminal.

 - Untar the files in the terminal present in the domino workspace by running the commands
 
        tar -xvzf pyAudioAnalysis.tar.gz

        tar -xvzf VoxCeleb1_mini.tar.gz
        
 - Activate environment
 
        source activate bob_kaldi36

 - Navigate to the pyAudioAnalysis directory and install some more dependencies
 
        cd pyAudioAnalysis
        pip install -e .

### 7. Extract MFCCs (Mel Frequency Cepstral Coefficients)
###### from dev dataset

        cd ../
        python extract_spkr_utterancewise_MFCCs.py --dataset dev

###### from test dataset

        python extract_spkr_utterancewise_MFCCs.py --dataset test      
       
### 8. Train the UBM 
##### (Universal Background Model) from the extracted MFCCs (dev dataset)

        python UBM_training.py
        
### 9. Train the iVector (extractor)

        python iVector_training.py
        
### 10. Extract iVectors
###### from dev dataset

        python extract_spkr_utterancewise_iVectors.py --dataset dev

###### from test dataset

        python extract_spkr_utterancewise_iVectors.py --dataset test

### 11. Evaluate model performance
#### False Positive Rate and False Negative Rate 
        python tpr_fpr.py
        
### 12. Download pre-trained model files trained larger set of speakers in VoxCeleb1 (iVector_v4.model,full_GMM-UBM_v4.model,diag_GMM-UBM_v4.model)
       wget https://dsp-workflow.s3-us-west-2.amazonaws.com/blog_bob_kaldi_speaker_verification/iVector_v4.model
       wget https://dsp-workflow.s3-us-west-2.amazonaws.com/blog_bob_kaldi_speaker_verification/full_GMM-UBM_v4.model
       wget https://dsp-workflow.s3-us-west-2.amazonaws.com/blog_bob_kaldi_speaker_verification/diag_GMM-UBM_v4.model
       
###### To measure the performance of the pre-trained model, simply update the config yaml file path of model_files, namely
######  diag_GMM-UBM : './model_files/diag_GMM-UBM_v4.model'
######  full_GMM-UBM : './model_files/full_GMM-UBM_v4.model'
######  iVector : './model_files/iVector_v4.model'
        
# References
https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html

https://www.idiap.ch/software/bob/docs/bob/bob.kaldi/master/py_api.html#module-bob.kaldi

https://www.researchgate.net/publication/268277174_Speaker_Verification_using_I-vector_Features
