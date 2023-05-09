import os
import sys
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))

from experiments.utils import constants

constants.create_dirs()