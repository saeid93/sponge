function upload_all(){
    gsutil cp -rn ~/inference_x/data/results gs://inference_x_results/
}

upload_all
