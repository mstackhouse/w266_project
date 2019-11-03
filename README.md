# w266_project
W266 Project Repository

Starting to build towards replicating the [CAML paper](https://github.com/jamesmullenbach/caml-mimic)

`constants.py` is pulled from their architecture to save some generic constant variables that will be used and allow you to reconfigure your directory pointers, etc.

`buildVocab.py` is the vocabulary builder. The vocab is also output into `./vocab/vocab.csv`, but note that we don't have at train/test/val split yet so this isn't the final vocab. This was moreso to get an idea of our corpus. 

`assemble_data.py` builds the source dataset from the raw archives, but also produces a second post processed dataset that converts symptom text to lists and combines all VAERS_IDs that should be merged, resulting in a single label vector with all applicable labels to that specific case report. Additional duplicate occurences of SYMPTOM TEXT are also removed.

`datafuncs.py` is a helper module that can contain functions necessary for working with our datasets. The function `read_data()` will read in the post processed dataset and convert the list columns into list objects, as they're originally imported as strings. 

Don't push notebooks to the repo for now since they're so unstable with git. Add to .gitignore. Enter code into a script (ideally modeled off of the CAML repo's architecture) and push that to the repo instead. 

