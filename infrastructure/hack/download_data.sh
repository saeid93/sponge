#!/bin/bash

function download_data(){
    # download experiment logs
    gsutil cp -rn gs://ipa-results-1/results.zip ~/sponge/data
    unzip ~/sponge/data/results.zip
    mv results ~/sponge/data
    rm ~/sponge/data/results.zip

    # download ml models
    gsutil cp -rn 'gs://ipa-models/myshareddir/torchhub' /mnt/myshareddir
    # gsutil cp -rn 'gs://ipa-models/myshareddir/huggingface' /mnt/myshareddir

    # download lstm trained model
    gsutil cp -r gs://ipa-models/lstm ~/sponge/data
}

download_data
