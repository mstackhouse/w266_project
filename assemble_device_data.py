'''
assemble_device_data.py
    Data assembly from device data sources
    Intended to serve as additional embedding data
    Data can be downloaded from https://open.fda.gov/downloads/
    under Device Adverse Events

'''
from tqdm import tqdm
import os
import zipfile
import shutil
import json
import pandas as pd
from buildVocab import CustAnalyzer

def extract_text(filename):
    ''' Extract event text from the medical device event extract data'''
    
    events = []
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        
    for result in event['results']:
        for text in result['mdr_text']:
            events += [text['text']]

    return events

# Loop the RAW_DIR
all_events = []
device_data_dir = "/home/stack/Documents/datasets/DEVICE/" 
if not os.path.exists:
    dev_data_dir = input("Location of device data: ")

# Build tokenizer
tokenize = CustAnalyzer(mask_dates=True, max_length=None)

# Loop the zip archives
for archive in tqdm(os.listdir(device_data_dir)):

    # Unzip the archive to subfolder tmp/
    with zipfile.ZipFile(device_data_dir + archive) as zf:
        zf.extractall(device_data_dir + 'tmp/')
    
    # In case there are multiple files in the archive loop over
    # the tmp folder
    for file in os.listdir(device_data_dir + 'tmp/'):
        tmp = extract_text(device_data_dir + 'tmp/' + file)
        tmp = [' '.join(tokenize(x)) for x in tmp]
        all_events += tmp
        
    # Remove the archive folder
    shutil.rmtree(device_data_dir + 'tmp/')  

# Output
pd.DataFrame({'TEXT': all_events}).to_csv(f'{DATA_DIR}device_data.csv')