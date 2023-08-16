function upload_metasereis(){
    gsutil cp -rn ~/ipa-private/results/final/metaseries/$1 gs://ipa-results/results/final/metaseries/$1
}

upload_metasereis
