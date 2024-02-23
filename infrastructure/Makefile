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



all: repos kubernetes gcloud utilities storage download_data gurobi install_kube_dev sync_time_main
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
	$(repos_script) PROJECT=$(PROJECT)
	@echo "repos.sh completed"

.PHONY: gcloud
gcloud:
	chmod +x $(gcloud_script)
	bash $(gcloud_script)
	@echo "gcloud.sh completed"

.PHONY: kubernetes
kubernetes:
	chmod +x $(kubernetes_script)
	bash $(kubernetes_script) K8SVERSION=$(K8SVERSION) GPU=$(GPU) VPABRANCH=$(VPABRANCH)
	@echo "kubernetes.sh completed"

utilities:
	chmod +x $(utilities_script)
	bash $(utilities_script) K8SVERSION=$(K8SVERSION) VPABRANCH=$(VPABRANCH)
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

# add remote_kubectl_install laod-tester

# sync_time_1:
# 	chmod +x $(sync_time_1)
# 	bash $(sync_time_1)
# 	@echo "sync_time_1.sh completed"

# sync_time_2:
# 	chmod +x $(sync_time_2)
# 	bash $(sync_time_2)
# 	@echo "sync_time_2.sh completed"

.PHONY: sync_time_main
sync_time_main:
	chmod +x $(sync_time_main)
	bash $(sync_time_main)
	@echo "sync_time_main.sh completed"

sync_time_load_tester:
	chmod +x $(sync_time_load_tester)
	bash $(sync_time_load_tester) PRIMARY_IP=$(PRIMARY_IP)
	@echo "sync_time_load_tester.sh completed"

.PHONY: install-seldon-v2
install-seldon-v2: install-microk8s ## Install seldon-v2
	make -C seldon-v2

##@ Tools

.PHONY: install-go
install-go:
	wget https://golang.org/dl/go$(GO_VERSION).linux-amd64.tar.gz
	sudo tar -C /usr/local -xzf go$(GO_VERSION).linux-amd64.tar.gz
	rm go$(GO_VERSION).linux-amd64.tar.gz
	echo 'export PATH=$$PATH:/usr/local/go/bin' | tee -a ~/.profile ~/.zshrc
	# echo 'export GOPATH=$$HOME/go' >> ~/.zshrc
	# echo 'export PATH=$(PATH):$GOPATH/bin' >> ~/.zshrc
	#source ~/.profile
	#source ~/.zshrc
	# go version

.PHONY: install-microk8s
install-microk8s:
	sudo snap install microk8s --classic

	# add microk8s group to the current user groups
	sudo usermod -a -G microk8s $$USER

	# change the owner of the kubernetes configuration files to the current user
	sudo chown -f -R $$USER ~/.kube || true

	# reload currnet user groups
	#newgrp microk8s

	microk8s status --wait-ready

	microk8s enable dashboard dns registry istio

	# MAKE EXTERNAL IP POSSIBLE WHEN SOEM SERVICES REQUEST IT
	microk8s enable metallb
	# Enter each IP address range delimited by comma (e.g. '10.64.140.43-10.64.140.49,192.168.0.105-192.168.0.111'): 192.168.0.105-192.168.0.111

	# microk8s kubectl get all --all-namespaces

	# make `kubectl` and `helm` commands available as microk8s aliases
	@echo "\n\033[1;36mTry to add kubectl='microk8s kubectl' and helm='microk8s helm' to the ~/.bash_aliases\033[0m\n"

	@if ! grep -q "Microk8s aliases" ~/.bash_aliases; then \
		echo "\n# Microk8s aliases\n\
		alias kubectl='microk8s kubectl'\n\
		alias helm='microk8s helm'\n\
		alias ks='kubectl -n seldon-mesh'\n\
		alias kds='kubectl describe -n seldon-mesh'\n\
		alias kls='kubectl logs -n seldon-mesh'" >> ~/.bash_aliases \
		@echo "\n\033[1;32mThe ~/.bash_aliases has been updated successfully!\033[0m\n"; \
    else \
        echo "\n\033[1;32mAliases already exists in .bash_aliases\033[0m\n"; \
    fi
