# w266_project
W266 Project Repository

Starting to build towards replicating the [CAML paper](https://github.com/jamesmullenbach/caml-mimic)

`constants.py` is pulled from their architecture to save some generic constant variables that will be used and allow you to reconfigure your directory pointers, etc.

`buildVocab.py` is the vocabulary builder. The vocab is also output into `./vocab/vocab.csv`, but note that we don't have at train/test/val split yet so this isn't the final vocab. This was moreso to get an idea of our corpus. 

Don't push notebooks to the repo for now since they're so unstable with git. Add to .gitignore. Enter code into a script (ideally modeled off of the CAML repo's architecture) and push that to the repo instead. 

