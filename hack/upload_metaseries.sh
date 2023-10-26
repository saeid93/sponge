function upload_metasereis(){
    gsutil cp -rn ~/ipa-private/data/results/final/metaseries/$SERIES gs://ipa-results/results/final/metaseries/$SERIES
}

upload_metasereis
