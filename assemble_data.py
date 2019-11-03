'''
prep_data.py
    Data assembly from raw sources
'''
import os
import pandas as pd
import zipfile
import shutil
from constants import *

if not os.path.isdir(RAW_DIR):
    RAW_DIR = input("Raw data directory: ")

# Loop the RAW_DIR
if not os.path.isfile(f'{DATA_DIR}{SOURCE_DATASET}'):
    for archive in os.listdir(RAW_DIR):
        print(f'Processing {archive}...')
        
        # Unzip the archive to subfolder tmp/
        with zipfile.ZipFile(RAW_DIR + archive) as zf:
            zf.extractall(RAW_DIR + 'tmp/')

        # Pull out the archive name
        sample = archive[:-13]
            
        # Read in the verbatim data
        verbatim = pd.read_csv(f'{RAW_DIR}tmp/{sample}VAERSDATA.csv', 
                            encoding='latin1', error_bad_lines=False)\
                            [['VAERS_ID', 'SYMPTOM_TEXT']]
        # Read in the coded data
        coded = pd.read_csv(f'{RAW_DIR}tmp/{sample}VAERSSYMPTOMS.csv', 
                            encoding='latin1', error_bad_lines=False)
        # Merge the two
        df = verbatim.merge(coded, how='left', on='VAERS_ID')
        
        # Assign sample for tracing
        df['SOURCE'] = sample

        # If the compiled dataset exists then stack, else form
        try:
            compiled = pd.concat([df, compiled])
        except:
            compiled = df

        # Remove the archive folder
        shutil.rmtree(RAW_DIR + 'tmp/')

    # Output the dataset
    print(f'Writing data to {DATA_DIR}{SOURCE_DATASET}')
    compiled.to_csv(f'{DATA_DIR}{SOURCE_DATASET}', index=False)

else:
    print('Skipping data assembled since file already exists')
    compiled = pd.read_csv(f'{DATA_DIR}{SOURCE_DATASET}', low_memory=False)

def join_symptoms(row):
    ''' Turn symptoms into a list in a single var '''
    out = []
    for i in range(1,6):
        if not pd.isna(row[f'SYMPTOM{i}']):
            out.append(row[f'SYMPTOM{i}'])
    return out

def join_versions(row):
    ''' Turns versions into a list in a single var '''
    out = []
    for i in range(1,6):
        if not pd.isna(row[f'SYMPTOMVERSION{i}']):
            out.append(row[f'SYMPTOMVERSION{i}'])
    return out

print('Post-processing...')
# Map the listing functions
compiled['SYMPTOMS'] = compiled.apply(join_symptoms,axis=1)
compiled['VERSIONS'] = compiled.apply(join_versions,axis=1)

# Get rid of the numbered variables and keep the subset
compiled = compiled[['VAERS_ID', 'SYMPTOM_TEXT', 'SYMPTOMS', 'VERSIONS']]

# Split off symptoms to merge into
source = compiled[['VAERS_ID', 'SYMPTOM_TEXT']].copy()
source.drop_duplicates(inplace=True) # We have duplicates here so remove

print('Aggregating lists...')
# Aggregate the code list
codes = compiled.groupby(['VAERS_ID'])['SYMPTOMS'].agg('sum').reset_index()
# Aggregate the version lists
versions = compiled.groupby(['VAERS_ID'])['VERSIONS'].agg('sum').reset_index()

# Merge into the final dataset
final = source.merge(codes, on='VAERS_ID')\
              .merge(versions, on='VAERS_ID')

print(f'Final dataset assembled. {final.shape[0]} records total.')
print(f'Writing to {DATA_DIR}post_processed.csv...')
final.to_csv(f'{DATA_DIR}post_processed.csv')
print('Done.')
