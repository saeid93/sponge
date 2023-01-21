import os
from .obj import setup_obj_store

# defined by the user
PROJECT_PATH = "/home/cc/infernece-pipeline-joint-optimization"
OBJ_PATH = "/home/cc/my_mounting_point/" # object store path

# base DATA folder path and object sore path
DATA_PATH = os.path.join(PROJECT_PATH, "data")
OBJ_DATA_PATH = os.path.join(OBJ_PATH, "data")
PIPELINES_FOLDER = "mlserver-prototype"
# configs path
PIPLINES_PATH = os.path.join(
    PROJECT_PATH, "pipelines", PIPELINES_FOLDER)
CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
PROFILING_CONFIGS_PATH = os.path.join(CONFIGS_PATH, "profiling")
NODE_PROFILING_CONFIGS_PATH = os.path.join(
    PROFILING_CONFIGS_PATH, "nodes")
PIPELINE_PROFILING_CONFIGS_PATH = os.path.join(
    PROFILING_CONFIGS_PATH, "pipelines")

# triton folders
TRITON_PROFILING_PATH = os.path.join(
    PROFILING_CONFIGS_PATH, "triton")
TRITON_PROFILING_CONFIGS_PATH = os.path.join(
    TRITON_PROFILING_PATH, "configs")
TRITON_PROFILING_TEMPLATES_PATH = os.path.join(
    TRITON_PROFILING_PATH, "templates")

# results noraml path
RESULTS_PATH = os.path.join(DATA_PATH, "results")
PROFILING_RESULTS_PATH = os.path.join(
    RESULTS_PATH, "profiling")
NODE_PROFILING_RESULTS_PATH = os.path.join(
    PROFILING_RESULTS_PATH, "nodes")
NODE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    NODE_PROFILING_RESULTS_PATH, "static")
NODE_PROFILING_RESULTS_TRITON_PATH = os.path.join(
    NODE_PROFILING_RESULTS_PATH, "triton")
NODE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    NODE_PROFILING_RESULTS_PATH, "dynamic")
PIPELINE_PROFILING_RESULTS_PATH = os.path.join(
    PROFILING_RESULTS_PATH, "pipelines")
PIPELINE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    PIPELINE_PROFILING_RESULTS_PATH, "static")
PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    PIPELINE_PROFILING_RESULTS_PATH, "dynamic")

# results object storage path
OBJ_RESULTS_PATH = os.path.join(OBJ_DATA_PATH, "results")
OBJ_PROFILING_RESULTS_PATH = os.path.join(
    OBJ_RESULTS_PATH, "profiling")
OBJ_NODE_PROFILING_RESULTS_PATH = os.path.join(
    OBJ_PROFILING_RESULTS_PATH, "nodes")
OBJ_NODE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    OBJ_NODE_PROFILING_RESULTS_PATH, "static")
OBJ_NODE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    OBJ_NODE_PROFILING_RESULTS_PATH, "dynamic")
OBJ_PIPELINE_PROFILING_RESULTS_PATH = os.path.join(
    OBJ_PROFILING_RESULTS_PATH, "pipelines")
OBJ_PIPELINE_PROFILING_RESULTS_STATIC_PATH = os.path.join(
    OBJ_PIPELINE_PROFILING_RESULTS_PATH, "static")
OBJ_PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH = os.path.join(
    OBJ_PIPELINE_PROFILING_RESULTS_PATH, "dynamic")

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
    if not os.path.exists(NODE_PROFILING_RESULTS_TRITON_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_TRITON_PATH)
    if not os.path.exists(NODE_PROFILING_RESULTS_DYNAMIC_PATH):
        os.makedirs(NODE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH):
        os.makedirs(PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(PIPELINE_PROFILING_RESULTS_STATIC_PATH):
        os.makedirs(PIPELINE_PROFILING_RESULTS_STATIC_PATH)
    if not os.path.exists(OBJ_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_RESULTS_PATH)
    if not os.path.exists(OBJ_PROFILING_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_PROFILING_RESULTS_PATH)
    if not os.path.exists(OBJ_NODE_PROFILING_RESULTS_PATH):
        setup_obj_store()
        os.makedirs(OBJ_NODE_PROFILING_RESULTS_PATH)
    if not os.path.exists(OBJ_NODE_PROFILING_RESULTS_STATIC_PATH):
        setup_obj_store()
        os.makedirs(OBJ_NODE_PROFILING_RESULTS_STATIC_PATH)
    if not os.path.exists(OBJ_NODE_PROFILING_RESULTS_DYNAMIC_PATH):
        setup_obj_store()
        os.makedirs(OBJ_NODE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(OBJ_PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH):
        setup_obj_store()
        os.makedirs(OBJ_PIPELINE_PROFILING_RESULTS_DYNAMIC_PATH)
    if not os.path.exists(OBJ_PIPELINE_PROFILING_RESULTS_STATIC_PATH):
        setup_obj_store()
        os.makedirs(OBJ_PIPELINE_PROFILING_RESULTS_STATIC_PATH)
        
# _create_dirs()

# prometheus client
PROMETHEUS = "http://localhost:30090"
