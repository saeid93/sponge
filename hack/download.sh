# considering that you have already connected to the object storage
function download(){
    gsutil cp -rn gs://ipa-results/results ~/ipa/data
}

download