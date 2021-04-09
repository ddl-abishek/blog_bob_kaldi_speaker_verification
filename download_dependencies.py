import os

# the script downloads pyAudioAnalysis from the S3 bucket. Although this library can be downlaoded throjugh pip install, specidfic changes havebeen made in this library
# for the sake of this project. Also a small subset of the dataset VoxCeleb1 (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) of the '/dev' set and '/test' set is downloaded.


os.system('wget https://dsp-workflow.s3-us-west-2.amazonaws.com/VoxCeleb1_mini.tar.gz')
os.system('wget https://dsp-workflow.s3-us-west-2.amazonaws.com/pyAudioAnalysis.tar.gz')