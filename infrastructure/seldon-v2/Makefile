SELDON_NAMESPACE ?= seldon-mesh
SELDON_WORKING_DIR ?= $(shell pwd)/seldon-core
SELDON_VERSION ?= 

SELDON_KUBECTL_NAMESPACE := --namespace $(SELDON_NAMESPACE)

SAMPLE_MODELS_DIR ?= $(SELDON_WORKING_DIR)/samples/models
SAMPLE_PIPELINES_DIR ?= $(SELDON_WORKING_DIR)/samples/pipelines

GO_VERSION ?= 1.20

KUBECTL = microk8s kubectl
HELM = microk8s helm

.PHONY: all
all: install-seldon-core-v2

##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

.PHONY: install-seldon-core-v2
install-seldon-core-v2: uninstall-seldon-core-v2 install-kafka ## uninstall the old version, and install the new one
	$(HELM) repo add seldon-charts https://seldonio.github.io/helm-charts
	$(HELM) repo update seldon-charts

	# install crds
	$(HELM) upgrade --install seldon-core-v2-crds seldon-charts/seldon-core-v2-crds
	
	# check the crds are up
	sleep 1
	$(KUBECTL) wait --for condition=established --timeout=60s crd/servers.mlops.seldon.io
	$(KUBECTL) wait --for condition=established --timeout=60s crd/serverconfigs.mlops.seldon.io
	
	# create namespace
	$(KUBECTL) create namespace $(SELDON_NAMESPACE) || echo "namespace already existing"

	# install core components
	$(HELM) upgrade --install seldon-core-v2 seldon-charts/seldon-core-v2-setup \
		$(SELDON_KUBECTL_NAMESPACE) --values components-values.yaml

	# install runtimes to install seldon on different namespaces
	$(HELM) upgrade --install seldon-v2-runtime seldon-charts/seldon-core-v2-runtime \
		$(SELDON_KUBECTL_NAMESPACE) --set config.tracingConfig.disable=true

	# install servers
	$(HELM) upgrade --install seldon-v2-servers seldon-charts/seldon-core-v2-servers \
		$(SELDON_KUBECTL_NAMESPACE)


.PHONY: uninstall-seldon-core-v2
uninstall-seldon-v2-core: ## Uninstall the old seldon-v2
	$(HELM) uninstall seldon-v2-servers $(SELDON_KUBECTL_NAMESPACE) || true
	$(HELM) uninstall seldon-v2-runtime $(SELDON_KUBECTL_NAMESPACE) || true
	$(HELM) uninstall seldon-core-v2 $(SELDON_KUBECTL_NAMESPACE) || true
	$(HELM) uninstall seldon-core-v2-crd || true

.PHONY: install-kafka
install-kafka: uninstall-kafka-dev uninstall-kafka ## Install the kafka using strimzi
	$(KUBECTL) delete all --all $(SELDON_KUBECTL_NAMESPACE)
	
	$(KUBECTL) apply -f https://strimzi.io/install/latest?namespace=$(SELDON_NAMESPACE) \
		$(SELDON_KUBECTL_NAMESPACE)
	
	# TODO: wget the https://strimzi.io/examples/latest/kafka/kafka-ephemeral.yaml and then
	# change the name to 'seldon'
	$(KUBECTL) apply -f kafka.yaml $(SELDON_KUBECTL_NAMESPACE)

	$(KUBECTL) wait --for=condition=Ready --timeout=300s kafka/seldon $(SELDON_KUBECTL_NAMESPACE)

.PHONY: uninstall-kafka
uninstall-kafka:
	$(KUBECTL) delete --ignore-not-found=true -f https://strimzi.io/install/latest?namespace=$(SELDON_NAMESPACE) \
		--wait=true $(SELDON_KUBECTL_NAMESPACE)
	$(KUBECTL) delete --ignore-not-found=true -f https://strimzi.io/examples/latest/kafka/kafka-ephemeral.yaml \
		--wait=true $(SELDON_KUBECTL_NAMESPACE)

.PHONY: install-kafka-dev
install-kafka-dev: pull-seldon-repo uninstall-kafka ## Install the kafka using strimzi
	$(HELM) repo add strimzi https://strimzi.io/charts/
	$(HELM) repo update strimzi

	$(HELM) upgrade --install strimzi-kafka-operator \
		strimzi/strimzi-kafka-operator \
		$(SELDON_KUBECTL_NAMESPACE) --create-namespace \
		--set featureGates='+UseKRaft\,+KafkaNodePools' \
		--set strimzi.kafkaNodePools.enabled=true

	$(HELM) upgrade --install seldon-core-v2-kafka $(SELDON_WORKING_DIR)/kafka/strimzi $(SELDON_KUBECTL_NAMESPACE)

.PHONY: uninstall-kafka-dev
uninstall-kafka-dev:
	$(HELM) uninstall seldon-core-v2-kafka $(SELDON_KUBECTL_NAMESPACE) || true
	$(HELM) uninstall strimzi-kafka-operator $(SELDON_KUBECTL_NAMESPACE) || true


.PHONY: install-seldon-cli
install-seldon-cli: pull-seldon-repo ## Local seldon cli build (requires Go)
	make -C $(SELDON_WORKING_DIR)/operator build-seldon
	set-seldon-cli-envs

.PHONY: pull-seldon-repo
pull-seldon-repo: ## Pull the seldon GitHub Repo
	git clone https://github.com/SeldonIO/seldon-core --branch=v2 --single-branch || true

.PHONY: set-seldon-cli-envs
set-seldon-cli-envs: ## Prepare seldon cli envs in the path
	@if ! grep -q "Seldon CLI" ~/.profile; then \
		echo "\n# Seldon CLI\n\
		export PATH=$$PATH:$(SELDON_WORKING_DIR)/operator/bin\n\n\
		export SELDON_INFER_HOST=$$($(KUBECTL) get svc seldon-mesh -n seldon-mesh -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):80\n\
		export SELDON_SCHEDULE_HOST=$$($(KUBECTL) get svc seldon-scheduler -n seldon-mesh -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9004\n\
		export SELDON_KAFKA_BROKER=seldon-kafka-bootstrap.seldon-mesh:9092" | tee -a ~/.profile ~/.zshrc \
		update-env-hint; \
	else \
        echo "\nSeldon CLI environments already exist in .profile\n";\
    fi

.PHONY: update-env-hint
update-env-hint: ## show to the user that should run a command
	@echo "\033[1;32mThe ~/.profile and ~/.zshrc files have been updated successfully!\033[0m\n"
	@echo "\033[1;33mPlease run 'source ~/.profile' or 'source ~/.zshrc' to set new environment.\033[0m\n"
