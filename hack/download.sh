# considering that you have already connected to the object storage
function download(){
    gsutil cp -rn gs://inference_x_results/results ~/inference_x/data
}

download