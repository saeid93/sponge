import os


# dfined by the user
DATA_PATH = "/Users/saeid/Codes/smart-kube/data"
CONFIGS_PATH = "/Users/saeid/Codes/smart-kube/smart-kube/configs"

# generated baesd on the users' path
WORKLOADS_PATH = os.path.join(DATA_PATH, "workloads")
RESULTS_PATH = os.path.join(DATA_PATH, "results")
ARABESQUE_PATH = os.path.join(DATA_PATH, "arabesque-raw")
ALIBABA_PATH = os.path.join(DATA_PATH, "alibaba-raw")
ANALYSIS_CONTAINERS_PATH = os.path.join(
    DATA_PATH, "analysis", "containers")
ANALYSIS_CLUSTERS_PATH = os.path.join(
    DATA_PATH, "analysis", "clusters")
FINAL_STATS_PATH = os.path.join(
    DATA_PATH, "final_stats"
)
