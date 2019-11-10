'''
prep_data.py
    Data assembly from raw sources
'''
import os
import pandas as pd
import zipfile
import shutil
from constants import *
from sklearn.model_selection import ShuffleSplit

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

# Join the lists together
final['SYMPTOMS'] = final.SYMPTOMS.apply(lambda x: ';'.join(x))
final['VERSIONS'] = final.VERSIONS.apply(lambda x: ';'.join([str(y) for y in x]))

# Write out
print(f'Final dataset assembled. {final.shape[0]} records total.')
print(f'Writing to {DATA_DIR}post_processed.csv...')
final.to_csv(f'{DATA_DIR}post_processed.csv')

# Create splits
print('Creating Train/Test/Val splits...')

# Create suffle split object
splitter = ShuffleSplit(train_size=TRAIN_PROP, 
                        test_size=TEST_PROP + VAL_PROP, 
                        random_state=1234, 
                        n_splits=1)

# Split out the indices
for train, test in splitter.split(final):
    train_ind = train
    devval_ind = test

# Relative proportions of dev/val 
test_prop = TEST_PROP / (TEST_PROP + VAL_PROP)
val_prop = VAL_PROP / (TEST_PROP + VAL_PROP)

# Split the test set from above again to get the dev and val sets
splitter = ShuffleSplit(train_size=test_prop, 
                        test_size=val_prop, 
                        random_state=1234, 
                        n_splits=1)

# Split out the last two sets
for test, val in splitter.split(devval_ind):
    test_ind = devval_ind[test]
    dev_ind = devval_ind[val]

# Write out the datasets
final.iloc[train_ind].to_csv(f'{DATA_DIR}/train.csv', index=False)
final.iloc[test_ind].to_csv(f'{DATA_DIR}/test.csv', index=False)
final.iloc[dev_ind].to_csv(f'{DATA_DIR}/dev.csv', index=False)

