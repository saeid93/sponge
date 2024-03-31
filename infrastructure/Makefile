current_dir = $(shell pwd)
hack_dir := $(current_dir)/hack
zsh_script := $(hack_dir)/zsh.sh
repos_script := $(hack_dir)/repos.sh
kubernetes_script := $(hack_dir)/kubernetes.sh
gcloud_script := $(hack_dir)/gcloud.sh
utilities_script := $(hack_dir)/utilities.sh
dev_utilities := $(hack_dir)/dev_utilities.sh
storage_script := $(hack_dir)/storage.sh
gurobi_script := $(hack_dir)/gurobi.sh
docker_installation := $(hack_dir)/docker_installation.sh
download_data := $(hack_dir)/download_data.sh
download_datasets := $(hack_dir)/download_datasets.sh
install_kube_dev := $(hack_dir)/install_kube_dev.sh
load_tester_key := $(hack_dir)/load_tester_key.sh
enable_external_kubectl := $(hack_dir)/enable_external_kubectl.sh
remote_kubectl := $(hack_dir)/remote_kubectl.sh
sync_time_1 := $(hack_dir)/sync_time_1.sh
sync_time_2 := $(hack_dir)/sync_time_2.sh
sync_time_load_tester := $(hack_dir)/sync_time_load_tester.sh
sync_time_main := $(hack_dir)/sync_time_main.sh



all: repos docker_installation kubernetes gcloud utilities storage download_data gurobi
	@echo "Installation of all packages and dependencies completed!"


.PHONY: dev-tools
dev-tools: kubernetes gcloud dev_utilities install_kube_dev
	@echo "Installation of all packages and dependencies completed!"

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

load_tester: repos gcloud
	@echo "Installation of load tester dependencies finished"

.PHONY: zsh
zsh:
	chmod +x $(zsh_script)
	$(zsh_script)
	@echo "zsh.sh completed"

.PHONY: repos
repos:
	chmod +x $(repos_script)
	$(repos_script)
	@echo "repos.sh completed"

.PHONY: gcloud
gcloud:
	chmod +x $(gcloud_script)
	bash $(gcloud_script)
	@echo "gcloud.sh completed"

.PHONY: kubernetes
kubernetes:
	chmod +x $(kubernetes_script)
	bash $(kubernetes_script) K8SVERSION=$(K8SVERSION) GPU=$(GPU)
	@echo "kubernetes.sh completed"

.PHONY: docker_installation
docker_installation:
	chmod +x $(docker_installation)
	bash $(docker_installation)
	@echo "docker_installation.sh completed"

utilities:
	chmod +x $(utilities_script)
	bash $(utilities_script) K8SVERSION=$(K8SVERSION)
	@echo "utilities.sh completed"

.PHONY: dev_utilities
dev_utilities:
	chmod +x $(utilities_script)
	bash $(dev_utilities_script)
	@echo "dev_utilities.sh completed"

.PHONY: gurobi
gurobi:
	chmod +x $(gurobi_script)
	bash $(gurobi_script)
	@echo "gurobi.sh completed"

.PHONY: storage
storage:
	chmod +x $(storage_script)
	bash $(storage_script) PUBLIC_IP=$(PUBLIC_IP)
	@echo "storage.sh completed"

.PHONY: download_data
download_data:
	chmod +x $(download_data)
	$(download_data)
	@echo "download_data.sh completed"

download_datasets:
	chmod +x $(download_datasets)
	$(download_datasets)
	@echo "download_datasets.sh completed"

.PHONY: install_kube_dev
install_kube_dev:
	chmod +x $(install_kube_dev)
	bash $(install_kube_dev)
	@echo "install_kube_dev.sh completed"

generate_and_show_key:
	if [ ! -f ~/.ssh/id_rsa.pub ]; then \
		echo y | ssh-keygen -f ~/.ssh/id_rsa; \
	fi
	cat ~/.ssh/id_rsa.pub

load_tester_key:
	chmod +x $(load_tester_key)
	bash $(load_tester_key)
	@echo "load_tester_key.sh completed"

enable_external_kubectl:
	chmod +x $(enable_external_kubectl)
	bash $(enable_external_kubectl) PUBLIC_IP=$(PUBLIC_IP)
	@echo "enable_external_kubectl.sh completed"

enable_ssh_and_kubectl: load_tester_key enable_external_kubectl PUBLIC_IP=$(PUBLIC_IP)

gcloud_login:
	bash gcloud auth login --no-launch-browser

# add merge of load_tester_key and enable_external_kubectl steps for server side operations as main side

remote_kubectl_install:
	chmod +x $(remote_kubectl)
	bash $(remote_kubectl) REMOTE_IP=$(REMOTE_IP)
	@echo "connect_to_remote.sh completed"
