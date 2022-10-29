import os

# defined by the user
PROJECT_PATH = "/home/cc/infernece-pipeline-joint-optimization"

# base DATA folder path
DATA_PATH = os.path.join(PROJECT_PATH, "data")

# configs path
PIPLINES_PATH = os.path.join(
    PROJECT_PATH, "pipelines", "mlserver-prototype")
CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
PROFILING_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "profiling")
NODE_PROFILING_CONFIGS_PATH = os.path.join(PROFILING_CONFIGS_PATH, "nodes")
PIPELINE_PROFILING_CONFIGS_PATH = os.path.join(
    PROFILING_CONFIGS_PATH, "pipelines")

# results path
RESULTS_PATH = os.path.join(DATA_PATH, "results")
PROFILING_RESULTS_PATH = os.path.join(RESULTS_PATH, "profiling")
NODE_PROFILING_RESULTS_PATH = os.path.join(
    PROFILING_RESULTS_PATH, "nodes")
NODE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    NODE_PROFILING_RESULTS_PATH, "static")
NODE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    NODE_PROFILING_RESULTS_PATH, "dynamic")
PIPELINE_PROFILING_RESULTS_PATH = os.path.join(
    PROFILING_RESULTS_PATH, "pipelines")
PIPELINE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    PIPELINE_PROFILING_RESULTS_PATH, "static")
PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    PIPELINE_PROFILING_RESULTS_PATH, "dynamic")

# generated baesd on the users' path
# TODO completely polish and remove unecessary ones
TEMP_MODELS_PATH = os.path.join(DATA_PATH, "model-temp")
KUBE_YAMLS_PATH = os.path.join(DATA_PATH, "yamls")
PIPELINES_MODELS = os.path.join(DATA_PATH, "pipeline-test-meta")

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(TEMP_MODELS_PATH):
        os.makedirs(TEMP_MODELS_PATH)
    if not os.path.exists(KUBE_YAMLS_PATH):
        os.makedirs(KUBE_YAMLS_PATH)
    if not os.path.exists(PIPELINES_MODELS):
        os.makedirs(PIPELINES_MODELS)
    if not os.path.exists(PIPLINES_PATH):
        os.makedirs(PIPLINES_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(PROFILING_CONFIGS_PATH):
        os.makedirs(PROFILING_CONFIGS_PATH)
    if not os.path.exists(NODE_PROFILING_CONFIGS_PATH):
        os.makedirs(NODE_PROFILING_CONFIGS_PATH)
    if not os.path.exists(PIPELINE_PROFILING_CONFIGS_PATH):
        os.makedirs(PIPELINE_PROFILING_CONFIGS_PATH)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    if not os.path.exists(PROFILING_RESULTS_PATH):
        os.makedirs(PROFILING_RESULTS_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_STATIC_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_STATIC_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_DYNAMIC_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH):
        os.makedirs(PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(PIPELINE_PROFILING_RESULTS_STATIC_PATH):
        os.makedirs(PIPELINE_PROFILING_RESULTS_STATIC_PATH)
        
_create_dirs()

# prometheus client
PROMETHEUS = "http://localhost:30090"
