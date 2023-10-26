.PHONY: clean_cluster download fetch_pipeline upload_all upload_metaseries

hack_dir := $(HOME)/ipa-private/hack
clean_cluster := $(hack_dir)/clean_cluster.sh
download := $(hack_dir)/download.sh
fetch_pipeline := $(hack_dir)/fetch_pipeline.sh
upload_all := $(hack_dir)/upload_all.sh
upload_metaseries := $(hack_dir)/upload_metaseries.sh

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

upload_all:
	chmod +x $(upload_all)
	$(upload_all)
	@echo "upload_all.sh completed"

upload_metaseries: SERIES ?= 
upload_metaseries: upload_metaseries_target

upload_metaseries_target:
	chmod +x $(upload_metaseries)
	bash $(upload_metaseries) $(SERIES)
	@echo "upload_metaseries.sh completed"

