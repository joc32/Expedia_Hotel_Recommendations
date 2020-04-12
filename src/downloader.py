import os
import zipfile


os.system("pip3 install kaggle")
os.system("kaggle competitions download -c expedia-hotel-recommendations -p ../datasets")

with zipfile.ZipFile(os.path.join('..','datasets','expedia-hotel-recommendations.zip'), 'r') as zip_ref:
    zip_ref.extractall(os.path.join('..','datasets'))

print('\n DATASET READY!!! ')