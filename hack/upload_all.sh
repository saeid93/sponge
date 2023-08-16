function upload_all(){
    gsutil cp -rn ~/ipa-private/data/results gs://ipa-results/
}

upload_all
