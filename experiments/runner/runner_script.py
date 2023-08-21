import subprocess
import os
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config-name", help="Path to configuration file", required=True)
args = parser.parse_args()

project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))
from experiments.utils.constants import PROJECT_PATH

runner_folder = os.path.join(PROJECT_PATH, "experiments", "runner")
script1_path = os.path.join(runner_folder, "experiments_runner.py")
time.sleep(10)
script2_path = os.path.join(runner_folder, "adaptation_runner.py")
def run_script(script_path, config_name):
    return subprocess.Popen(["python", script_path, "--config-name", config_name])

config_name = args.config_name
process1 = run_script(script1_path, config_name)
process2 = run_script(script2_path, config_name)

while True:
    if process1.poll() is not None and process2.poll() is not None:
        break
    time.sleep(1)

sys.exit(0)
