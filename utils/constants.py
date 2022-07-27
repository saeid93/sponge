import os

# dfined by the user
DATA_PATH = "/home/cc/infernece-pipeline-joint-optimization/data"

# generated baesd on the users' path
TEMP_MODELS_PATH = os.path.join(DATA_PATH, "model-temp")
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "train-results")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "test-results")
KUBE_YAMLS_PATH = os.path.join(DATA_PATH, "yamls")
TRITON_CONFIG_PATH = os.path.join(DATA_PATH, "triton-configs")

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(TEMP_MODELS_PATH):
        os.makedirs(TEMP_MODELS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)
    if not os.path.exists(TESTS_RESULTS_PATH):
        os.makedirs(TESTS_RESULTS_PATH)
    if not os.path.exists(KUBE_YAMLS_PATH):
        os.makedirs(KUBE_YAMLS_PATH)
    if not os.path.exists(TRITON_CONFIG_PATH):
        os.makedirs(TRITON_CONFIG_PATH)

_create_dirs()