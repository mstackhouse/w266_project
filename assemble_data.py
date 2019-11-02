'''
prep_data.py
    Data assembly from raw sources
'''
import os
import pandas as pd
import zipfile
import shutil

rootdir = '/media/stack/Storage/Documents/datasets/VAERS/'
if not os.path.isdir(rootdir):
    rootdir = input("Data directory: ")

# Loop the rootdir
for archive in os.listdir(rootdir):
    print(f'Processing {archive}...')
    
    # Unzip the archive to subfolder tmp/
    with zipfile.ZipFile(rootdir + archive) as zf:
        zf.extractall(rootdir + 'tmp/')

    # Pull out the archive name
    sample = archive[:-13]
        
    # Read in the verbatim data
    verbatim = pd.read_csv(f'{rootdir}tmp/{sample}VAERSDATA.csv', 
                           encoding='latin1', error_bad_lines=False)\
                           [['VAERS_ID', 'SYMPTOM_TEXT']]
    # Read in the coded data
    coded = pd.read_csv(f'{rootdir}tmp/{sample}VAERSSYMPTOMS.csv', 
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
    shutil.rmtree(rootdir + 'tmp/')

# Output the dataset
print(f'Writing data to {rootdir}/w266_full_data.csv')
compiled.to_csv(f'{rootdir}/w266_full_data.csv')

print('Done.')
