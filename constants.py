PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 100
MAX_LENGTH = 2500

TRAIN_PROP = 0.8
TEST_PROP = 0.15
VAL_PROP = 0.05

assert sum([TRAIN_PROP, TEST_PROP, VAL_PROP]) == 1.0,\
    "Train/Test/Val split proportions must sum to 1."

#where you want to save any models you may train
MODEL_DIR = '/path/to/repo/saved_models/'

RAW_DIR = '/media/stack/Storage/Documents/datasets/VAERS/'
DATA_DIR = './vaersdata/'
VOCAB_DIR = './vocab/'
LABEL_DIR = './labels/'
SOURCE_DATASET = 'w266_full_data.csv'
TRAINING_DATA = 'train.csv'
SUPPLEMENTAL_DATA = 'supplement_data.txt'