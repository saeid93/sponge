import os
import sys

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from experiments.utils.constants import (
    OBJ_PATH
)

from experiments.utils.obj import (
    setup_obj_store
)

setup_obj_store()

# TODO get dataset of object detector
a = 1
# TODO get dataset of object classifier
# TODO get dataset of audio to text
# TODO get dataset of question answering
# TODO get dataset of text summariser
# TODO get dataset of language identification
# TODO get dataset of neural machine translation