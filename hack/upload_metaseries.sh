function upload_metasereis(){
    gsutil cp -rn ~/inference_x/results/final/metaseries/$1 gs://inference_x_results/results/final/metaseries/$1
}

upload_metasereis
