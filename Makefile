.PHONY: clean_cluster download fetch_pipeline upload_all upload_metaseries

hack_dir := $(HOME)/ipa-private/hack
clean_cluster := $(hack_dir)/clean_cluster.sh
download := $(hack_dir)/download.sh
fetch_pipeline := $(hack_dir)/fetch_pipeline.sh
upload_all_results := $(hack_dir)/upload_all_results.sh
upload_metaseries := $(hack_dir)/upload_metaseries.sh
upload_all_models := $(hack_dir)/upload_all_models.sh
upload_node_models := $(hack_dir)/upload_node_models.sh

clean_cluster:
	chmod +x $(clean_cluster)
	$(clean_cluster)
	@echo "clean_cluster.sh completed"

clean_cluster_force:
	chmod +x $(clean_cluster)
	$(clean_cluster) --force
	@echo "clean_cluster.sh completed"

download:
	chmod +x $(download)
	$(download)
	@echo "download.sh completed"

fetch_pipeline:
	chmod +x $(fetch_pipeline)
	$(fetch_pipeline)
	@echo "fetch_pipeline.sh completed"

upload_all_results:
	chmod +x $(upload_all_results)
	$(upload_all_results)
	@echo "upload_all_results.sh completed"

upload_metaseries: SERIES ?= 
upload_metaseries: upload_metaseries_target

upload_metaseries_target:
	chmod +x $(upload_metaseries)
	bash $(upload_metaseries) $(SERIES)
	@echo "upload_metaseries.sh completed"

upload_all_models:
	chmod +x $(upload_all_models)
	$(upload_all_models)
	@echo "upload_all_models.sh completed"

upload_node_models: NODE ?=
upload_node_models: SOURCE ?= 
upload_node_models: upload_node_models_target

upload_node_models_target:
	chmod +x $(upload_node_models)
	bash $(upload_node_models) $(SOURCE) $(NODE)
	@echo "upload_node_models.sh completed"

lint:
	black --check .
	# flake8 .
	# Check if something has changed after generation
	git \
		--no-pager diff \
		--exit-code \
		.