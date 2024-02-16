# Training hyperparameters
##TODO fix all this

LEARNING_RATE = 0.001
BATCH_SIZE = 4
MIN_EPOCHS = 3
MAX_EPOCHS =50

# Dataset
TRAIN_PATH = 'v1.1/splits/flood_handlabeled/flood_train_data.csv'
VALID_PATH = 'v1.1/splits/flood_handlabeled/flood_valid_data.csv'
TEST_PATH = 'v1.1/splits/flood_handlabeled/flood_test_data.csv'
NUM_WORKERS = 4

# Compute related
ACCELERATOR = "gpu"
PRECISION = 16