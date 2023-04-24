import subprocess
import os
import sys
import time

# TODO unifying reading yaml config files for both scripts here here

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..", "..")))
from experiments.utils.constants import PROJECT_PATH

runner_folder = os.path.join(PROJECT_PATH, "experiments", "runner")

# Define the paths to the two Python script files
script1_path = os.path.join(runner_folder, "experiments_runner.py")
time.sleep(10)
script2_path = os.path.join(runner_folder, "adaptation_runner.py")


# Define a function to run each script in a separate subprocess
def run_script(script_path):
    return subprocess.Popen(["python", script_path])


# Start the two subprocesses
process1 = run_script(script1_path)
process2 = run_script(script2_path)

# Wait for both subprocesses to complete
while True:
    if process1.poll() is not None and process2.poll() is not None:
        break
    time.sleep(1)

# Exit the parent process
sys.exit(0)
