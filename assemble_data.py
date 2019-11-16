'''
assemble_data.py
    Data assembly from raw sources
'''
import os
import pandas as pd
import zipfile
import shutil
from sklearn.model_selection import ShuffleSplit
from constants import *
from buildVocab import CustAnalyzer

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

print('Post-processing...')
# Map the listing functions
compiled['SYMPTOMS'] = compiled.apply(join_symptoms,axis=1)

# Get rid of the numbered variables and keep the subset
compiled = compiled[['VAERS_ID', 'SYMPTOM_TEXT', 'SYMPTOMS']]

# Split off symptoms to merge into
source = compiled[['VAERS_ID', 'SYMPTOM_TEXT']].copy()
source.drop_duplicates(inplace=True) # We have duplicates here so remove

# Pre-process the symptom text
tokenize = CustAnalyzer(mask_dates=True)
raw_tokenize = CustAnalyzer(mask_dates=True, max_length=None)

source['TEXT'] = source.SYMPTOM_TEXT.apply(lambda x: ' '.join(tokenize(x)))
source['RAW_TEXT'] = source.SYMPTOM_TEXT.apply(lambda x: ' '.join(raw_tokenize(x)))

print('Aggregating labels...')
# Aggregate the code list
codes = compiled.groupby(['VAERS_ID'])['SYMPTOMS'].agg('sum').reset_index()

# Merge into the final dataset
final = source.merge(codes, on='VAERS_ID')

# Join the lists together
final['LABELS'] = final.SYMPTOMS.apply(lambda x: ';'.join(x))

# Keep only the variables I want
final = final[['VAERS_ID', 'TEXT', 'LABELS', 'RAW_TEXT']]

# Write out
print(f'Final dataset assembled. {final.shape[0]} records total.')
print(f'Writing to {DATA_DIR}post_processed.csv...')
final.to_csv(f'{DATA_DIR}post_processed.csv', index=False)

# Record length for sorting purposes
final['length'] = final.TEXT.apply(
    lambda x: len(x.split() if not pd.isnull(x) else 0)
    )

# Filter out empty case reports
final = final[final.length > 1]
# Filter out missing labels
final = final[pd.notna(final.LABELS)]

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

# Spit out the raw training records for the vocab build and embedding
# training
final[['RAW_TEXT']].iloc[train_ind]\
    .to_csv(f'{DATA_DIR}/train_raw.csv')
final.drop(['RAW_TEXT'], axis=1, inplace=True)

# Write out the datasets
train = final.iloc[train_ind]\
    .sort_values(['length'])
train.to_csv(f'{DATA_DIR}/train.csv')

test = final.iloc[test_ind]\
    .sort_values(['length'])
test.to_csv(f'{DATA_DIR}/test.csv')

dev = final.iloc[dev_ind]\
    .sort_values(['length'])
dev.to_csv(f'{DATA_DIR}/dev.csv')

print('Done.')